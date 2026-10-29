# ============================================================
#  Maestro Experiment Lab — tiny fast A/B training on Kaggle
#  One variant per GPU, fixed token budget, blind-sample zips,
#  everything published to the public HF experiments repo.
#
#  usage:
#    python train_experiment.py prep              # build + upload data packs (once)
#    python train_experiment.py run <variant>     # control | muon | data | both
#
#  Author: Vaibhav Agarwal (@thisisforlearn) - sole author; final authority on this project
#  License: GPL-3.0-only <https://www.gnu.org/licenses/gpl-3.0.html>
# ============================================================
import os, sys, time, json, glob, shutil, zipfile, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from rich.console import Console
from musicbot.model import GPT, GEN_CFG
from musicbot.dataset import download_maestro, download_mutopia, tokenize_corpus, PackedData
from musicbot.train import pretrain
from musicbot.generate import compose, decode_tokens
from musicbot import synth

REPO = "neko-i-am-a-cat/maestro-experiments"   # public HF dataset repo
GCFG = dict(GEN_CFG, d=768, n_layer=12, n_head=12, ff=2048, ctx=1024, drop=0.1)
CTX = 1024
STEPS_DEFAULT = 1500            # fixed token budget => fair A/B (same tokens seen)
BS = 12
LR = 3e-4
SEEDS = [11, 22, 33]            # fixed sample seeds across ALL variants
SAMPLE_MIN = 0.66               # ~40 s per sample
TRAIN_SECONDS_CAP = 58 * 60     # safety net if a step is slow
GIANT_FOLDER = "https://drive.google.com/drive/folders/1Stz3CAvMoplo79LR5I3onMWRelCugBYS"
SF3_URL = "https://huggingface.co/datasets/mileslilly/soundfonts/resolve/main/MuseScore_General.sf3"

con = Console(highlight=False)


def hf():
    tok = os.environ.get("HF_TOKEN")
    if not tok:
        try:
            from kaggle_secrets import UserSecretsClient
            tok = UserSecretsClient().get_secret("HF_TOKEN")
        except Exception:
            tok = None
    if not tok:
        raise SystemExit("HF_TOKEN not found (add it to Kaggle Secrets)")
    from huggingface_hub import HfApi
    api = HfApi(token=tok)
    api.create_repo(REPO, repo_type="dataset", exist_ok=True, private=False)
    return api


def hf_up(api, path, name, log=print):
    api.upload_file(path_or_fileobj=str(path), path_in_repo=name, repo_id=REPO,
                    repo_type="dataset")
    log(f"uploaded: {name}")


def hf_down(name, dest_dir, log=print):
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(repo_id=REPO, filename=name, repo_type="dataset", local_dir=str(dest_dir))
    return Path(p)


# ---------------------------------------------------------------- data packs
def _tokenize(root_list, out_dir, log=print):
    out_dir.mkdir(parents=True, exist_ok=True)
    workers = max(2, (os.cpu_count() or 4) // 2)  # halve cores: two lab jobs share a session
    tokenize_corpus([str(r) for r in root_list], str(out_dir), workers=workers, log=log)
    return out_dir


def cmd_prep(work):
    """Build pack_base (MAESTRO+Mutopia) and pack_giant (GiantMIDI), upload both."""
    api = hf()
    packs = work / "packs"
    # ---- base pack
    base_dir = packs / "base"
    if not (base_dir / "packed.npy").exists():
        con.print("[bold]building base pack (MAESTRO + Mutopia)...[/]")
        maestro = download_maestro(str(work / "dl"), log=con.print)
        mutopia = download_mutopia(str(work / "dl"), log=con.print)
        _tokenize([maestro, mutopia], base_dir, log=con.print)
    meta = json.loads((base_dir / "meta.json").read_text())
    con.print(f"base pack: {meta['n_tokens']:,} tokens / {meta['n_pieces']} pieces")
    hf_up(api, base_dir / "packed.npy", "base/packed.npy", con.print)
    hf_up(api, base_dir / "meta.json", "base/meta.json", con.print)
    # ---- giant pack
    giant_dir = packs / "giant"
    if not (giant_dir / "packed.npy").exists():
        con.print("[bold]downloading GiantMIDI-Piano (193 MB)...[/]")
        raw = work / "giantmidi_raw"
        raw.mkdir(parents=True, exist_ok=True)
        r = subprocess.run([sys.executable, "-m", "gdown", "--folder", GIANT_FOLDER,
                            "-O", str(raw), "--remaining-ok", "--fuzzy"],
                           capture_output=True, text=True)
        mids = list(raw.rglob("*.mid")) + list(raw.rglob("*.midi"))
        for z in list(raw.rglob("*.zip")) + list(raw.rglob("*.tar.gz")):
            shutil.unpack_archive(str(z), str(raw / "unpacked"))
        mids += list((raw / "unpacked").rglob("*.mid")) + list((raw / "unpacked").rglob("*.midi"))
        if not mids:
            con.print("[bold red]GiantMIDI download failed — skip data variants this round[/]")
            return
        con.print(f"GiantMIDI: {len(mids)} MIDI files")
        _tokenize([Path(m) for m in mids][:8000], giant_dir, log=con.print)
        meta = json.loads((giant_dir / "meta.json").read_text())
        con.print(f"giant pack: {meta['n_tokens']:,} tokens / {meta['n_pieces']} pieces")
    hf_up(api, giant_dir / "packed.npy", "giant/packed.npy", con.print)
    hf_up(api, giant_dir / "meta.json", "giant/meta.json", con.print)
    con.print("[bold green]prep done[/] — packs are on the hub for every future session")


def get_corpus(variant, work):
    """corpus for a variant: giant pack for data/both, base otherwise; local fallback."""
    d = work / f"corpus_{variant}"
    want_giant = variant in ("data", "both")
    names = (["giant/packed.npy", "giant/meta.json"] if want_giant
             else ["base/packed.npy", "base/meta.json"])
    try:
        d.mkdir(parents=True, exist_ok=True)
        for n in names:
            hf_down(n, d, con.print)
        src = d / names[0].split("/")[0]
        return PackedData(src, CTX)
    except Exception as e:
        con.print(f"[yellow]pack download failed ({e}) — tokenizing locally[/]")
    base = work / "corpus_local"
    if not (base / "packed.npy").exists():
        maestro = download_maestro(str(work / "dl"), log=con.print)
        mutopia = download_mutopia(str(work / "dl"), log=con.print)
        _tokenize([maestro, mutopia], base, log=con.print)
    return PackedData(base, CTX)


# ---------------------------------------------------------------- training
def build_opt(variant, model):
    if variant in ("muon", "both"):
        from musicbot.muon import split_params, HybridOpt
        muon, adam = split_params(model, muon_lr=0.02, adam_lr=LR)
        n_mu = sum(p.numel() for g in muon.param_groups for p in g["params"])
        n_ad = sum(p.numel() for g in adam.param_groups for p in g["params"])
        con.print(f"Muon on {n_mu / 1e6:.1f}M hidden params · AdamW on {n_ad / 1e6:.1f}M")
        return HybridOpt(muon, adam)
    return None  # pretrain builds AdamW


def setup_sound(work):
    sf_dir = ROOT / "soundfonts"
    sf_dir.mkdir(exist_ok=True)
    sf = sf_dir / "MuseScore_General.sf3"
    if not sf.exists():
        subprocess.run(["curl", "-sL", "-o", str(sf), SF3_URL], check=False)
    return sf


def cmd_run(variant, work, steps):
    assert variant in ("control", "muon", "data", "both"), variant
    api = hf()
    dev = "cuda"
    torch.manual_seed(0)
    data = get_corpus(variant, work)
    con.print(f"corpus: {data.n_tokens:,} tokens / {data.n_pieces} pieces")
    gen = GPT(**GCFG).to(dev)
    n_params = sum(p.numel() for p in gen.parameters())
    con.print(f"model: {n_params / 1e6:.1f}M params · variant [bold]{variant}[/] · {steps} steps")

    losses = []

    def ckpt(model, step, ema):
        losses.append((step, round(float(ema_loss[0]), 4)))

    ema_loss = [0.0]
    t0 = time.time()

    def log(line):
        con.print(line)
        try:
            ema_loss[0] = float(line.split("loss ")[1].split()[0])
        except Exception:
            pass

    _, ema = pretrain(gen, data, steps, bs=BS, lr=LR, device=dev, bf16=True,
                      amp_device="cuda", log=log, ckpt_fn=ckpt, ckpt_every=250,
                      max_seconds=TRAIN_SECONDS_CAP, use_bar=True, ls=0.1,
                      opt=build_opt(variant, gen))
    wall = time.time() - t0
    ema.copy_to(gen)

    # ---------------- samples (fixed seeds = paired comparison across variants)
    setup_sound(work)
    out = work / f"samples_{variant}"
    out.mkdir(exist_ok=True)
    results = dict(variant=variant, steps=steps, params_M=round(n_params / 1e6, 1),
                   corpus_tokens=int(data.n_tokens), wall_min=round(wall / 60, 1),
                   final_loss=round(ema_loss[0], 4), seeds=SEEDS,
                   date=time.strftime("%Y-%m-%d %H:%M"))
    for s in SEEDS:
        try:
            mid = out / f"s{s}.mid"
            toks = compose(gen, None, minutes=SAMPLE_MIN, K=1, device=dev, precision="fp32",
                           seed=s, log=con.print)
            decode_tokens(toks).save(str(mid))
            wav = out / f"s{s}.wav"
            synth.render_wav(mid, wav, log=con.print)
            mp3 = out / f"s{s}.mp3"
            r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav),
                                "-b:a", "128k", str(mp3)], capture_output=True)
            if r.returncode == 0 and mp3.exists():
                wav.unlink()
            results[f"s{s}_notes"] = mid.exists()
        except Exception as e:
            con.print(f"[yellow]sample seed {s} failed: {e}[/]")
    (out / "results.json").write_text(json.dumps(results, indent=2))
    con.print(json.dumps(results, indent=2))

    # ---------------- publish: blind zip + checkpoint + results
    zname = f"E_{variant}_{time.strftime('%m%d_%H%M')}.zip"
    zpath = work / zname
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(out.iterdir()):
            z.write(f, arcname=f"{variant}/{f.name}")
    hf_up(api, zpath, f"samples/{zname}", con.print)
    hf_up(api, out / "results.json", f"results/{variant}_{steps}.json", con.print)
    ck = work / f"ckpt_{variant}.pt"
    torch.save({"gen_cfg": GCFG, "gen": gen.state_dict(),
                "meta": {"variant": variant, "steps": steps, "author": "Vaibhav Agarwal",
                         "license": "GPL-3.0-only"}}, ck)
    hf_up(api, ck, f"ckpt/ckpt_{variant}_{steps}.pt", con.print)
    con.print(f"[bold green]variant [white]{variant}[/] published to hf.co/datasets/{REPO}[/]")


if __name__ == "__main__":
    work = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("lab_work")
    work.mkdir(exist_ok=True, parents=True)
    if len(sys.argv) < 2 or sys.argv[1] == "prep":
        cmd_prep(work)
    elif sys.argv[1] == "run":
        variant = sys.argv[2] if len(sys.argv) > 2 else "control"
        steps = int(sys.argv[3]) if len(sys.argv) > 3 else STEPS_DEFAULT
        cmd_run(variant, work, steps)
    else:
        raise SystemExit("usage: prep | run <control|muon|data|both> [steps]")
