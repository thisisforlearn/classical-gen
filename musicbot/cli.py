# ============================================================
#  Maestro CLI — rich terminal interface
#  Author: Vaibhav Agarwal (@thisisforlearn) - sole author; final authority on this project
#  License: GPL-3.0-only <https://www.gnu.org/licenses/gpl-3.0.html>
# ============================================================
from __future__ import annotations
import argparse, json, os, shutil, sys, time
from importlib import import_module
from pathlib import Path

import numpy as np
import torch
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn,
)
from rich.table import Table

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
OUT = ROOT / "output"
BUNDLE = MODELS / "maestro_bundle.pt"

c = Console()


def banner():
    c.print(
        Panel.fit(
            "[bold magenta]MAESTRO[/] — classical composer bot\n"
            "[dim]author [bold]Vaibhav[/] · GPL-3.0-only · v1.0.0[/]",
            border_style="magenta",
        )
    )


def save_bundle(path, gen, crit, gcfg, ccfg, meta):
    MODELS.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "gen_cfg": dict(gcfg),
            "crit_cfg": dict(ccfg),
            "gen": gen.state_dict() if gen is not None else None,
            "critic": crit.state_dict() if crit is not None else None,
            "meta": meta,
        },
        path,
    )


def cmd_check(_):
    t = Table(title="environment", box=box.SIMPLE)
    t.add_column("item"); t.add_column("value", style="cyan")
    t.add_row("python", sys.version.split()[0])
    t.add_row("torch", torch.__version__)
    t.add_row("cpu threads", str(torch.get_num_threads()))
    t.add_row("cpu capability", torch.backends.cpu.get_cpu_capability())
    t.add_row("bundle", "[green]present[/]" if BUNDLE.exists() else "[red]missing — train on kaggle first (`python -m musicbot kaggle`)[/]")
    if BUNDLE.exists():
        meta = torch.load(BUNDLE, map_location="cpu", weights_only=False).get("meta", {})
        t.add_row("bundle meta", json.dumps(meta, default=str)[:120])
    t.add_row("ffmpeg", shutil.which("ffmpeg") or "no (mp3 export disabled)")
    t.add_row("musescore", shutil.which("musescore") or "not found (pdf skipped, musicxml still written)")
    c.print(t)


def cmd_prepare(args):
    from musicbot.dataset import download_maestro, tokenize_corpus

    root = download_maestro(DATA, log=c.print)
    extra = []
    if args.mutopia:
        from musicbot.dataset import download_mutopia

        extra.append(str(download_mutopia(DATA, log=c.print)))
    meta = tokenize_corpus(root, DATA / "tokens", workers=args.workers, log=c.print, extra_roots=extra)
    c.print(f"[green]corpus ready:[/] {meta['n_pieces']} pieces · {meta['n_tokens'] / 1e6:.2f}M tokens")


def cmd_train(args):
    from musicbot.dataset import PackedData
    from musicbot.model import GPT, GEN_CFG, CRIT_CFG
    from musicbot.train import pretrain

    data = PackedData(DATA / "tokens", ctx=args.ctx)
    gcfg = dict(GEN_CFG)
    gcfg.update(ctx=args.ctx, d=args.dim, n_layer=args.layers)
    gen = GPT(gcfg)
    c.print(f"generator: [cyan]{gen.num_params() / 1e6:.1f}M[/] params · device cpu · bf16={args.bf16}")
    pretrain(gen, data, args.steps, args.bs, args.lr, "cpu", bf16=args.bf16, amp_device="cpu",
             log=c.print,
             ckpt_fn=lambda m, s: save_bundle(BUNDLE, m, None, gcfg, dict(CRIT_CFG), {"stage": "pretrain", "step": s}))
    save_bundle(BUNDLE, gen, None, gcfg, dict(CRIT_CFG), {"stage": "pretrain", "step": args.steps})
    c.print(f"[green]saved {BUNDLE}[/]")


def cmd_rl(args):
    from musicbot.dataset import PackedData
    from musicbot.model import GPT, Critic
    from musicbot.rl import train_critic, reinforce
    import copy

    b = torch.load(BUNDLE, map_location="cpu", weights_only=False)
    data = PackedData(DATA / "tokens", ctx=b["gen_cfg"]["ctx"])
    gen = GPT(b["gen_cfg"]); gen.load_state_dict(b["gen"])
    ref = copy.deepcopy(gen)
    crit = Critic(b["crit_cfg"]) if b.get("critic") else Critic()
    c.print("training critic (real vs generated)...")
    train_critic(crit, gen, data, args.critic_steps, 16, 3e-4, "cpu", log=c.print)
    c.print("reinforce fine-tuning...")
    reinforce(gen, ref, crit, data, args.rl_steps, "cpu", G=4, seg=192, log=c.print)
    save_bundle(BUNDLE, gen, crit, b["gen_cfg"], b.get("crit_cfg", dict(crit.cfg)), {"stage": "rl", "step": args.rl_steps})
    c.print(f"[green]saved {BUNDLE}[/]")


def cmd_kaggle(_):
    cell = build_kaggle_cell()
    out = MODELS / "kaggle_cell.py"
    MODELS.mkdir(exist_ok=True)
    out.write_text(cell)
    c.print(Panel(
        f"[bold]One-paste Kaggle cell written to:[/] {out}  ({len(out.read_text().splitlines())} lines)\n\n"
        f"[bold green]Option A — GitHub (recommended, always up to date):[/]\n"
        f"  On Kaggle (GPU T4 x2, Internet On) paste ONE cell:\n"
        f"    [cyan]!test -d classical-gen || git clone https://github.com/thisisforlearn/classical-gen[/]\n"
        f"    [cyan]!pip install -q mido rich[/]\n"
        f"    [cyan]!python classical-gen/kaggle/train_kaggle.py[/]\n"
        f"  Resume: Save Version -> in a new session attach the previous Output as Input;\n"
        f"  the script auto-detects ckpt.pt / maestro_bundle.pt and continues.\n\n"
        f"[bold yellow]Option B — one-paste fallback (no GitHub needed):[/]\n"
        f"  Open the file above -> Select All -> Copy -> paste into an empty Kaggle cell -> Run.\n\n"
        f"[cyan]Output:[/] maestro_bundle.zip + 3 preview MIDIs in the Output panel.\n"
        f"[cyan]Then locally:[/] unzip into {MODELS} -> [bold]python -m musicbot all[/]",
        title="kaggle training", border_style="cyan",
    ))


def build_kaggle_cell() -> str:
    header = (
        "# ============================================================\n"
        "#  MAESTRO one-paste Kaggle training cell\n"
        "#  Author: Vaibhav Agarwal (@thisisforlearn) - sole author; final authority on this project\n"
        "#  License: GPL-3.0-only <https://www.gnu.org/licenses/gpl-3.0.html>\n"
        "#  Trains generator + critic + RL on Kaggle GPU, exports maestro_bundle.zip\n"
        "# ============================================================\n"
    )
    order = ["tokenizer", "model", "dataset", "train", "rl", "generate"]
    prelude = (
        "import subprocess, sys\n"
        "subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'mido', 'rich'], check=False)\n"
        "print('deps ready')\n"
    )
    parts = [header, prelude]
    for name in order:
        text = Path(import_module(f"musicbot.{name}").__file__).read_text()
        parts.append(_flatten(text))
    driver = Path(ROOT / "kaggle" / "train_kaggle.py").read_text()
    parts.append(_flatten(driver))
    return "\n".join(parts)


def _flatten(src: str) -> str:
    lines = src.splitlines()
    out, i, n = [], 0, len(lines)
    while i < n:
        ln = lines[i]
        s = ln.strip()
        if s.startswith("from __future__"):
            i += 1
            continue
        if s.startswith("from musicbot."):
            if s.endswith("("):
                i += 1
                while i < n and (lines[i].startswith((" ", "\t")) or lines[i].strip() == ")"):
                    i += 1
            else:
                i += 1
            continue
        if s == "try:" and i + 1 < n and lines[i + 1].lstrip().startswith("from musicbot."):
            i += 1
            while i < n and (
                lines[i].lstrip().startswith("from musicbot.")
                or lines[i].startswith("        ")
                or lines[i].strip() == ")"
            ):
                i += 1
            if i < n and lines[i].strip().startswith("except"):
                i += 1
            while i < n and (
                lines[i].lstrip().startswith("from ")
                or lines[i].startswith("        ")
                or lines[i].strip() == ")"
            ):
                i += 1
            continue
        out.append(ln)
        i += 1
    return "\n".join(out)


def cmd_generate(args):
    from musicbot.generate import generate

    device = args.device
    if device == "auto":
        device = "cpu"
    OUT.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    midi = OUT / f"opus_{ts}.mid"
    generate(str(BUNDLE), str(midi), args.minutes, args.candidates, device, args.precision,
             args.seed, log=c.print, console=c)
    c.print(f"[green]MIDI ready:[/] {midi}")
    return midi


def cmd_render(args):
    from musicbot.synth import render_wav, maybe_mp3
    from musicbot.sheet import midi_to_musicxml, musicxml_to_pdf

    OUT.mkdir(exist_ok=True)
    midi = args.midi or sorted(OUT.glob("*.mid"))[-1]
    stem = Path(midi).stem
    if getattr(args, "orchestra", False):
        from musicbot.orchestrate import orchestrate
        from musicbot.synth import _find_sf2
        sf2 = _find_sf2() or ""
        presets = "sonatina" if "onatina" in Path(sf2).name.lower() else "gm"
        orch = OUT / f"{stem}_orch.mid"
        orchestrate(str(midi), str(orch), presets=presets, log=c.print)
        midi = orch
        stem = Path(midi).stem
    wav = OUT / f"{stem}.wav"
    render_wav(str(midi), str(wav), log=c.print)
    maybe_mp3(str(wav), OUT, log=c.print)
    xml = OUT / f"{stem}.musicxml"
    midi_to_musicxml(str(midi), str(xml), log=c.print)
    musicxml_to_pdf(str(xml), OUT, log=c.print)
    c.print(f"[green]rendered:[/] {wav.name} · {xml.name}")


def cmd_all(args):
    midi = cmd_generate(args)
    args.midi = str(midi)
    cmd_render(args)


def cmd_bench(args):
    import torch.nn as nn
    from torch.ao.quantization import quantize_dynamic
    from musicbot.dataset import PackedData
    from musicbot.model import GPT, Critic, GEN_CFG, CRIT_CFG
    from musicbot.generate import _gen_batch

    torch.set_num_threads(min(12, os.cpu_count() or 1))
    data = PackedData(DATA / "tokens", ctx=512)
    g = GPT(dict(GEN_CFG, d=384, n_layer=8, ff=1024, ctx=512))
    c.print(f"[cyan]{g.num_params() / 1e6:.1f}M[/] params · {torch.get_num_threads()} threads · fp32 · AVX2/oneDNN")
    opt = torch.optim.AdamW(g.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)
    rng = np.random.default_rng(0)
    with Progress(SpinnerColumn(), TextColumn("[bold cyan]cpu training  "),
                  BarColumn(bar_width=44), TextColumn("[green]{task.fields[m]}"),
                  TimeElapsedColumn(), TimeRemainingColumn(), console=c) as p:
        tk = p.add_task("bench", total=args.steps, m="")
        t0, toks = time.time(), 0
        for step in range(1, args.steps + 1):
            x, y = data.sample_batch(16, rng)
            _, loss = g(torch.from_numpy(x), torch.from_numpy(y))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(g.parameters(), 1.0)
            opt.step()
            toks += x.size
            p.update(tk, completed=step,
                     m=f"loss {loss.item():.3f} · {toks / (time.time() - t0) / 1e3:.1f}k tok/s")
    trate = toks / (time.time() - t0)
    c.print(f"[green]train:[/] {trate / 1e3:.1f}k tok/s → Kaggle-size job (26M tok × 3 epochs) ≈ "
            f"[bold]{26e6 * 3 / trate / 3600:.1f} h on this CPU[/] (vs ~1 h on T4)")
    MODELS.mkdir(exist_ok=True)
    torch.save({"gen_cfg": dict(GEN_CFG, d=384, n_layer=8, ff=1024, ctx=512), "crit_cfg": dict(CRIT_CFG),
                "gen": g.state_dict(), "critic": Critic().state_dict(), "meta": {"bench": True}},
               Path("/tmp/opencode/bench.pt"))
    gen_q, _, _ = None, None, None
    from musicbot.generate import load_bundle

    gen_q, _, _b = load_bundle("/tmp/opencode/bench.pt", "cpu", "int8", log=lambda s: None)
    prompts = torch.tensor([[1, 238, 4, 5, 124, 216, 180]] * 4, dtype=torch.long)
    total = 2 * 256 * 4
    with Progress(SpinnerColumn(), TextColumn("[bold cyan]int8 generate "),
                  BarColumn(bar_width=44), TextColumn("[green]{task.fields[m]}"),
                  TimeElapsedColumn(), TimeRemainingColumn(), console=c) as p:
        tk = p.add_task("bench", total=total, m="")
        t0, done = time.time(), 0
        for _ in range(2):
            out = _gen_batch(gen_q, prompts, 256, 0.95, 40, 0.95, 1.15, "cpu", False)
            done += out.numel()
            p.update(tk, completed=done, m=f"{done / (time.time() - t0):.0f} tok/s")
    grate = done / (time.time() - t0)
    c.print(f"[green]generate:[/] {grate:.0f} tok/s (best-of-4 batched) → 16-min piece ≈ [bold]{15000 / grate / 60:.1f} min[/] on CPU")


def cmd_windows(_):
    c.print(Panel(
        "[bold]Windows iGPU (Iris Xe) generation exe[/]\n\n"
        "[cyan]1.[/] Copy the whole music-bot folder to Windows (e.g. C:\\music-bot)\n"
        "[cyan]2.[/] Copy maestro_bundle.pt into C:\\music-bot\\models\\\n"
        "[cyan]3.[/] Open Command Prompt in C:\\music-bot\\windows and run:  [bold]build_exe.bat[/]\n"
        "      (installs torch-directml + builds MaestroComposer.exe with PyInstaller)\n"
        "[cyan]4.[/] Run:  [bold]dist\\MaestroComposer.exe[/]  -> composes on the Iris Xe iGPU\n"
        "      MIDI lands next to the exe. No Python needed afterwards.",
        title="windows exe", border_style="cyan",
    ))


def main():
    banner()
    p = argparse.ArgumentParser(prog="musicbot", description="Maestro classical composer (author: Vaibhav, GPL-3.0)")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check")
    pb = sub.add_parser("bench")
    pb.add_argument("--steps", type=int, default=20)
    pt = sub.add_parser("prepare"); pt.add_argument("--workers", type=int, default=None)
    pt.add_argument("--mutopia", action="store_true")
    ptr = sub.add_parser("train")
    ptr.add_argument("--steps", type=int, default=2000)
    ptr.add_argument("--bs", type=int, default=16)
    ptr.add_argument("--ctx", type=int, default=512)
    ptr.add_argument("--lr", type=float, default=3e-4)
    ptr.add_argument("--dim", type=int, default=384)
    ptr.add_argument("--layers", type=int, default=8)
    ptr.add_argument("--bf16", action="store_true", default=True)
    prl = sub.add_parser("rl-train")
    prl.add_argument("--critic-steps", type=int, default=300)
    prl.add_argument("--rl-steps", type=int, default=100)
    sub.add_parser("kaggle")
    pg = sub.add_parser("generate")
    pg.add_argument("--minutes", type=float, default=16.0)
    pg.add_argument("--candidates", type=int, default=4)
    pg.add_argument("--precision", choices=["int8", "bf16", "fp32"], default="int8")
    pg.add_argument("--device", default="auto")
    pg.add_argument("--seed", type=int, default=0)
    pren = sub.add_parser("render"); pren.add_argument("--midi", default=None)
    pren.add_argument("--orchestra", action="store_true")
    pall = sub.add_parser("all")
    pall.add_argument("--minutes", type=float, default=16.0)
    pall.add_argument("--candidates", type=int, default=4)
    pall.add_argument("--precision", choices=["int8", "bf16", "fp32"], default="int8")
    pall.add_argument("--device", default="auto")
    pall.add_argument("--seed", type=int, default=0)
    pall.add_argument("--orchestra", action="store_true")
    sub.add_parser("windows")
    a = p.parse_args()
    dict(check=cmd_check, prepare=cmd_prepare, train=cmd_train, bench=cmd_bench, **{"rl-train": cmd_rl},
         kaggle=cmd_kaggle, generate=cmd_generate, render=cmd_render, all=cmd_all,
         windows=cmd_windows)[a.cmd](a)


if __name__ == "__main__":
    main()
