# ============================================================
#  KAGGLE DRIVER — max-quality single-session training
#  Author: Vaibhav (sole author; final authority on this project)
#  License: GPL-3.0-only <https://www.gnu.org/licenses/gpl-3.0.html>
#
#  ~85M generator on both T4s (DataParallel, bf16, FlashAttention)
#  MAESTRO v3 + Mutopia corpus · critic · REINFORCE RL
#  Time-budgeted to fit ONE 12h Kaggle session with margin.
# ============================================================
import os, sys, subprocess, shutil, copy, time, json

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "mido", "rich"], check=False)

import numpy as np
import torch
from rich.console import Console
from rich.panel import Panel

con = Console()
con.print(Panel.fit("[bold magenta]MAESTRO Kaggle trainer — MAX QUALITY[/] · author Vaibhav · GPL-3.0-only"))
assert torch.cuda.is_available(), "Enable GPU accelerator: Settings -> Accelerator -> GPU T4 x2"
dev = "cuda"
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cuda.enable_flash_sdp(True)
con.print(f"GPU: [cyan]{torch.cuda.get_device_name(0)}[/] x{torch.cuda.device_count()}")

WORK = "/kaggle/working"
os.makedirs(WORK, exist_ok=True)

GCFG = dict(GEN_CFG, d=768, n_layer=12, n_head=12, ff=2048, ctx=1024)
CCFG = dict(CRIT_CFG, d=512, n_layer=8, n_head=8, ff=1536, ctx=512)

t_data = time.time()
root = download_maestro(WORK + "/data", log=con.print)
extra = []
try:
    extra.append(str(download_mutopia(WORK + "/data", log=con.print)))
except Exception as e:
    con.print(f"[yellow]Mutopia unavailable ({e}) — continuing with MAESTRO only[/]")
meta = tokenize_corpus(root, WORK + "/tokens", workers=4, log=con.print, extra_roots=extra)
data = PackedData(WORK + "/tokens", ctx=GCFG["ctx"])
con.print(f"corpus ready in {(time.time() - t_data) / 60:.1f} min: {meta['n_pieces']} pieces · "
          f"[cyan]{meta['n_tokens'] / 1e6:.1f}M tokens[/]")

gen = GPT(GCFG)
con.print(f"generator: [cyan]{gen.num_params() / 1e6:.1f}M[/] params · ctx {GCFG['ctx']} · 2xT4 bf16")


def _ck(model, step):
    torch.save({"gen_cfg": GCFG, "crit_cfg": CCFG, "gen": model.state_dict(), "critic": None,
                "meta": {"stage": "pretrain", "step": step}}, WORK + "/ckpt.pt")


resumed = False
if os.path.exists(WORK + "/ckpt.pt"):
    try:
        ck = torch.load(WORK + "/ckpt.pt", map_location="cpu", weights_only=False)
        if ck.get("meta", {}).get("stage") == "pretrain" and ck["gen_cfg"]["d"] == GCFG["d"]:
            gen.load_state_dict(ck["gen"])
            resumed = True
            con.print(f"[green]resumed from ckpt.pt at step {ck['meta']['step']}[/]")
    except Exception as e:
        con.print(f"[yellow]resume skipped: {e}[/]")

con.print(Panel("stage 1/3: generator pretraining (~6.5 h budget)", border_style="cyan"))
pretrain(gen, data, steps=20000, bs=12, lr=6e-4, device=dev, bf16=True, amp_device="cuda",
         log=con.print, ckpt_fn=_ck, ckpt_every=200, parallel=True, accum=2, max_seconds=6.5 * 3600)

con.print(Panel("stage 2/3: critic training (grader)", border_style="cyan"))
ref = copy.deepcopy(gen).eval()
for p in ref.parameters():
    p.requires_grad_(False)
crit = Critic(CCFG)
con.print(f"critic: [cyan]{crit.num_params() / 1e6:.1f}M[/] params")
train_critic(crit, gen, data, steps=1200, bs=16, lr=3e-4, device=dev, G=6, seg=256, log=con.print)

con.print(Panel("stage 3/3: REINFORCE fine-tuning from critic rewards", border_style="cyan"))
reinforce(gen, ref, crit, data, steps=600, device=dev, G=8, seg=256, lr=1e-5, beta=0.05, log=con.print)

con.print(Panel("exporting bundle", border_style="green"))
torch.save({"gen_cfg": GCFG, "crit_cfg": CCFG, "gen": gen.state_dict(), "critic": crit.state_dict(),
            "meta": {"stage": "complete", "tokens": meta, "finished": True}},
           WORK + "/maestro_bundle.pt")

with torch.no_grad():
    for s in range(3):
        seeds = torch.from_numpy(data.sample_batch(1, np.random.default_rng(s))[0][:, :96]).to(dev)
        sample = sample_continuations(gen, seeds, 768, temp=0.95, device=dev)
        decode_tokens(sample[0].tolist()).save(WORK + f"/preview_{s}.mid")

shutil.make_archive(WORK + "/maestro_bundle", "zip", WORK, "maestro_bundle.pt")

con.print(Panel(
    "[bold green]TRAINING COMPLETE[/]\n\n"
    "1. In the notebook Output panel download [bold]maestro_bundle.zip[/] (and the preview MIDIs to listen now)\n"
    "2. On your machine:\n"
    "     unzip maestro_bundle.zip -d ~/music-bot/models/\n"
    "3. Compose:  source ~/ml_env/bin/activate && cd ~/music-bot && python -m musicbot all\n\n"
    "[dim]If the session died mid-training: re-run the cell — it resumes from ckpt.pt[/]",
    title="next steps", border_style="green",
))
