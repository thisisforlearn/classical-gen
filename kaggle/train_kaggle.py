# ============================================================
#  Maestro — Kaggle training driver (package mode, git-clone flow)
#  Author: Vaibhav (sole author; final authority on this project)
#  License: GPL-3.0-only <https://www.gnu.org/licenses/gpl-3.0.html>
#  Run on Kaggle:  !git clone https://github.com/thisisforlearn/classical-gen
#                  !pip install -q mido rich
#                  !python classical-gen/kaggle/train_kaggle.py
#  Backup: checkpoints auto-upload to Hugging Face Hub
#  (huggingface.co/thisisforlearn/maestro-checkpoints) every ~1200 steps
#  IF a Kaggle secret named HF_TOKEN exists — every upload is a new revision,
#  so full checkpoint history is preserved.
#  Resume: on a fresh session the script auto-downloads the last ckpt.pt
#  from the Hub — no manual Input attachment needed.
# ============================================================
import os, sys, subprocess, glob, shutil, time, json, zipfile, urllib.request, urllib.parse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "mido", "rich", "huggingface_hub"],
               check=False)
import numpy as np
import torch
from rich.console import Console
from rich.panel import Panel
from musicbot.model import GPT, Critic, GEN_CFG, CRIT_CFG
from musicbot.dataset import download_maestro, download_mutopia, tokenize_corpus, PackedData
from musicbot.train import pretrain, EMA
from musicbot.rl import train_critic, reinforce
from musicbot.generate import compose

con = Console()
con.print(Panel.fit("[bold cyan]Maestro[/] — classical composer trainer\n"
                    "[dim]author: Vaibhav · GPL-3.0-only[/]", border_style="cyan"))
assert torch.cuda.is_available(), "GPU required — enable GPU (T4 x2) in Kaggle settings"
dev = "cuda"
ngpu = torch.cuda.device_count()
con.print(f"GPU: {torch.cuda.get_device_name(0)} x{ngpu} · bf16={torch.cuda.is_bf16_supported()}")
hf_ok = hf_check(hf_token(), log=con.print)

WORK = "/kaggle/working"
LOCAL = os.path.isdir(WORK) and os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None
OUT = WORK if LOCAL else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
DATA = "data" if LOCAL else os.path.join(os.path.dirname(OUT), "data")
os.makedirs(OUT, exist_ok=True)
os.makedirs(DATA, exist_ok=True)

GCFG = dict(GEN_CFG, d=768, n_layer=12, n_head=12, ff=2048, ctx=1024, drop=0.1)
CCFG = dict(CRIT_CFG, d=512, n_layer=8, n_head=8, ff=1536, ctx=512)
PRE_STEPS, PRE_SECONDS = 20000, 6.5 * 3600
CKPT = os.path.join(OUT, "ckpt.pt")
BUNDLE = os.path.join(OUT, "maestro_bundle.pt")
ZIP = os.path.join(OUT, "maestro_bundle.zip")
HF_REPO = "neko-i-am-a-cat/maestro-checkpoints"


def hf_token():
    tok = os.environ.get("HF_TOKEN")
    if tok:
        return tok.strip()
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret("HF_TOKEN").strip()
    except Exception:
        return None


def hf_check(tok, log=print):
    if not tok:
        log("[yellow]HF backup OFF — add a Kaggle secret named HF_TOKEN (Add-ons -> Secrets), "
            "token type 'Write'[/]")
        return False
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=tok)
        who = api.whoami()
        role = (who.get("auth", {}).get("accessToken", {}) or {}).get("role", "?")
        if role != "write":
            log(f"[yellow]HF token is '{role}' — needs a 'Write' token for uploads. "
                f"Create one: huggingface.co/settings/tokens[/]")
            return False
        api.create_repo(HF_REPO, repo_type="model", exist_ok=True)
        log(f"[green]HF OK — logged in as [bold]{who.get('name')}[/] · "
            f"backup repo live: huggingface.co/{HF_REPO}[/]")
        return True
    except Exception as e:
        log(f"[yellow]HF token test failed ({e}) — backup disabled, training continues[/]")
        return False


def hf_fetch(name, dest, log=print):
    try:
        from huggingface_hub import hf_hub_download
        p = hf_hub_download(HF_REPO, name, repo_type="model", token=hf_token() or None)
        shutil.copy(p, dest)
        log(f"[green]downloaded previous {name} from huggingface.co/{HF_REPO}[/]")
        return True
    except Exception:
        return False


def hf_backup(path, name, tok, log=print):
    if not tok:
        log("[yellow]backup skipped — add a Kaggle secret named HF_TOKEN (Add-ons -> Secrets)[/]")
        return False
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=tok)
        api.create_repo(HF_REPO, repo_type="model", exist_ok=True)
        api.upload_file(path_or_fileobj=path, path_in_repo=name, repo_id=HF_REPO,
                        repo_type="model",
                        commit_message=f"backup {name} · {time.strftime('%Y-%m-%d %H:%M')}")
        log(f"[green]backed up {name} to huggingface.co/{HF_REPO} "
            f"(~{os.path.getsize(path) / 1e6:.0f} MB)[/]")
        return True
    except Exception as e:
        log(f"[yellow]backup failed (training continues): {e}[/]")
        return False


def make_zip():
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_STORED) as z:
        for f in [BUNDLE] + sorted(glob.glob(os.path.join(OUT, "preview_*.mid"))):
            z.write(f, os.path.basename(f))


# ---------------- data ----------------
tok_dir = os.path.join(DATA, "tokens")
inputs = glob.glob("/kaggle/input/*/tokens/packed.npy") if LOCAL else []
if not os.path.exists(os.path.join(tok_dir, "packed.npy")) and inputs:
    shutil.copytree(os.path.dirname(inputs[0]), tok_dir, dirs_exist_ok=True)
    con.print("[green]reused tokenized corpus from attached previous output[/]")
if not os.path.exists(os.path.join(tok_dir, "packed.npy")):
    maestro_dir = download_maestro(DATA, log=con.print)
    mutopia_dir = download_mutopia(DATA, log=con.print)
    roots = [maestro_dir] + ([mutopia_dir] if mutopia_dir else [])
    tokenize_corpus(roots[0], tok_dir, extra_roots=roots[1:], workers=4, log=con.print)
data = PackedData(tok_dir, 1024)
con.print(f"corpus: [bold]{data.tokens.size // 10**6}M[/] tokens")

# ---------------- models ----------------
gen = GPT(GCFG)
crit = Critic(CCFG)
con.print(f"generator: [bold]{sum(p.numel() for p in gen.parameters()) / 1e6:.0f}M[/] params · "
          f"critic: {sum(p.numel() for p in crit.parameters()) / 1e6:.0f}M params")

# ---------------- resume detection ----------------
tok = hf_token() if hf_ok else None
prev_bundles = [BUNDLE] + (glob.glob("/kaggle/input/*/maestro_bundle.pt") if LOCAL else [])
prev_ckpt = [CKPT] + (glob.glob("/kaggle/input/*/ckpt.pt") if LOCAL else [])
if any(os.path.exists(p) for p in prev_bundles):
    src = next(p for p in prev_bundles if os.path.exists(p))
    if src != BUNDLE:
        shutil.copy(src, BUNDLE)
    con.print(Panel.fit("[bold green]complete bundle found — training already done![/]\n"
                        "Re-zipping outputs with fresh previews.", border_style="green"))
    make_zip()
    con.print(Panel.fit(f"[bold green]DONE[/] — download [bold]maestro_bundle.zip[/] from the Output panel\n"
                        f"then locally: unzip -> python -m musicbot all", border_style="green"))
    sys.exit(0)
if not any(os.path.exists(p) for p in prev_ckpt):
    if hf_fetch("ckpt.pt", CKPT, log=con.print):
        prev_ckpt = [CKPT]
if any(os.path.exists(p) for p in prev_ckpt):
    src = next(p for p in prev_ckpt if os.path.exists(p))
    sd = torch.load(src, map_location="cpu", weights_only=False)
    gen.load_state_dict(sd["gen"])
    ema = EMA(gen)
    ema.load(sd["ema"])
    step0 = int(sd.get("step", 0))
    con.print(f"[green]resumed from checkpoint at step {step0}[/]")
else:
    ema, step0 = EMA(gen), 0

_last_up = [0]

def _ck(model, step, em):
    torch.save({"gen": model.state_dict(), "ema": em.state(), "step": step, "cfg": GCFG}, CKPT)
    if step - _last_up[0] >= 1200 or step >= PRE_STEPS:
        if hf_backup(CKPT, "ckpt.pt", tok, log=con.print):
            _last_up[0] = step

# ---------------- stage 1: pretrain ----------------
if step0 < PRE_STEPS:
    con.print(Panel.fit(f"[bold]stage 1/3 — pretrain[/] {PRE_STEPS - step0} steps · budget {PRE_SECONDS / 3600:.1f}h"))
    gen, ema = pretrain(gen, data, PRE_STEPS, bs=12, lr=3e-4, device=dev, bf16=True,
                        log=con.print, ckpt_fn=_ck, ckpt_every=150, amp_device="cuda",
                        parallel=ngpu > 1, accum=2, max_seconds=PRE_SECONDS,
                        use_bar=True, ls=0.1)
    _ck(gen, PRE_STEPS, ema)
else:
    con.print("[green]pretrain already complete — skipping to critic[/]")

# ---------------- stage 2: critic ----------------
con.print(Panel.fit("[bold]stage 2/3 — critic training[/]"))
train_critic(crit, gen, data, steps=1200, bs=16, lr=3e-4, device=dev, G=6, seg=256,
             log=con.print, use_bar=True)

# ---------------- stage 3: REINFORCE ----------------
con.print(Panel.fit("[bold]stage 3/3 — REINFORCE fine-tuning[/]"))
ref = GPT(GCFG)
ref.load_state_dict(gen.state_dict())
reinforce(gen, ref, crit, data, steps=600, device=dev, G=8, seg=256, lr=1e-5, beta=0.05,
          log=con.print, use_bar=True)

# ---------------- export (EMA weights = better sampling) ----------------
con.print(Panel.fit("[bold]exporting[/] — EMA weights into bundle"))
ema.copy_to(gen)
torch.save({"gen_cfg": GCFG, "crit_cfg": CCFG, "gen": gen.state_dict(),
            "critic": crit.state_dict(),
            "meta": {"author": "Vaibhav", "license": "GPL-3.0-only",
                     "corpus_tokens": int(data.tokens.size), "steps": PRE_STEPS}}, BUNDLE)

con.print("generating 3 preview pieces (30s each)...")
for i in range(3):
    torch.manual_seed(2024 + i)
    compose(gen, crit, minutes=0.5, K=2, device=dev).save(os.path.join(OUT, f"preview_{i + 1}.mid"))
    con.print(f"  preview_{i + 1}.mid")

make_zip()
hf_backup(ZIP, "maestro_bundle.zip", tok, log=con.print)
con.print(Panel.fit("[bold green]DONE[/] — download [bold]maestro_bundle.zip[/] from the Output panel\n"
                    "then locally: unzip -> python -m musicbot all", border_style="green"))
