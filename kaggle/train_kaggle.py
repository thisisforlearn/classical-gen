# ============================================================
#  Maestro — Kaggle training driver (package mode, git-clone flow)
#  Author: Vaibhav (sole author; final authority on this project)
#  License: GPL-3.0-only <https://www.gnu.org/licenses/gpl-3.0.html>
#  Run on Kaggle:  !git clone https://github.com/thisisforlearn/classical-gen
#                  !pip install -q mido rich
#                  !python classical-gen/kaggle/train_kaggle.py
#  Resume: attach this notebook's previous Output as an Input dataset —
#  the script auto-detects ckpt.pt / maestro_bundle.pt there and continues.
# ============================================================
import os, sys, subprocess, glob, shutil, time, zipfile
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "mido", "rich"], check=False)
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

WORK = "/kaggle/working"
LOCAL = os.path.isdir("/kaggle/working") and os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None
OUT = WORK if LOCAL else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
DATA = os.path.join(OUT, "..", "data") if not LOCAL else "data"
os.makedirs(OUT, exist_ok=True)
os.makedirs(DATA, exist_ok=True)

GCFG = dict(GEN_CFG, d=768, n_layer=12, n_head=12, ff=2048, ctx=1024, drop=0.1)
CCFG = dict(CRIT_CFG, d=512, n_layer=8, n_head=8, ff=1536, ctx=512)
PRE_STEPS, PRE_SECONDS = 20000, 6.5 * 3600
CKPT = os.path.join(OUT, "ckpt.pt")
BUNDLE = os.path.join(OUT, "maestro_bundle.pt")

# ---------------- data ----------------
tok_dir = os.path.join(DATA, "tokens")
inputs = glob.glob("/kaggle/input/*/tokens/packed.npy") if LOCAL else []
if not os.path.exists(os.path.join(tok_dir, "packed.npy")) and inputs:
    shutil.copytree(os.path.dirname(inputs[0]), tok_dir, dirs_exist_ok=True)
    con.print("[green]reused tokenized corpus from attached previous output[/]")
if not os.path.exists(os.path.join(tok_dir, "packed.npy")):
    maestro_dir = download_maestro(DATA, log=con.print)
    mutopia_dir = download_mutopia(DATA, log=con.print)
    tokenize_corpus([maestro_dir, mutopia_dir] if mutopia_dir else [maestro_dir], tok_dir,
                    workers=4, log=con.print)
data = PackedData(tok_dir)
con.print(f"corpus: [bold]{data.tokens.size // 10**6}M[/] tokens")

# ---------------- models ----------------
gen = GPT(GCFG)
crit = Critic(CCFG)
con.print(f"generator: [bold]{sum(p.numel() for p in gen.parameters()) / 1e6:.0f}M[/] params · "
          f"critic: {sum(p.numel() for p in crit.parameters()) / 1e6:.0f}M params")

# ---------------- resume detection ----------------
prev_bundles = [BUNDLE] + (glob.glob("/kaggle/input/*/maestro_bundle.pt") if LOCAL else [])
prev_ckpt = [CKPT] + (glob.glob("/kaggle/input/*/ckpt.pt") if LOCAL else [])
if any(os.path.exists(p) for p in prev_bundles):
    src = next(p for p in prev_bundles if os.path.exists(p))
    if src != BUNDLE:
        shutil.copy(src, BUNDLE)
    con.print(Panel.fit("[bold green]complete bundle found — training already done![/]\n"
                        "Re-run outputs maestro_bundle.zip with fresh previews.", border_style="green"))
elif any(os.path.exists(p) for p in prev_ckpt):
    src = next(p for p in prev_ckpt if os.path.exists(p))
    sd = torch.load(src, map_location="cpu", weights_only=False)
    gen.load_state_dict(sd["gen"])
    ema = EMA(gen)
    ema.load(sd["ema"])
    step0 = int(sd.get("step", 0))
    con.print(f"[green]resumed from checkpoint at step {step0}[/]")
else:
    ema, step0 = EMA(gen), 0

def _ck(model, step, em):
    torch.save({"gen": model.state_dict(), "ema": em.state(), "step": step, "cfg": GCFG}, CKPT)

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
    mid = os.path.join(OUT, f"preview_{i + 1}.mid")
    compose(gen, crit, minutes=0.5, K=2, device=dev).save(mid)
    con.print(f"  preview_{i + 1}.mid")

if os.path.exists(CKPT):
    os.remove(CKPT)
zip_path = os.path.join(OUT, "maestro_bundle.zip")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as z:
    for f in [BUNDLE] + sorted(glob.glob(os.path.join(OUT, "preview_*.mid"))):
        z.write(f, os.path.basename(f))
con.print(Panel.fit("[bold green]DONE[/] — download [bold]maestro_bundle.zip[/] from the Output panel\n"
                    "then locally: unzip → python -m musicbot all", border_style="green"))
