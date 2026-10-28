# ============================================================
#  Maestro — Windows iGPU (Iris Xe) composer
#  Author: Vaibhav (sole author; final authority on this project)
#  License: GPL-3.0-only <https://www.gnu.org/licenses/gpl-3.0.html>
# ============================================================
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from rich.console import Console
from rich.panel import Panel

c = Console()
c.print(Panel.fit("[bold magenta]MAESTRO[/] Windows composer · author Vaibhav · GPL-3.0-only"))

device = "cpu"
backend = "cpu (AVX2)"
try:
    import torch_directml

    device = torch_directml.device()
    backend = f"DirectML iGPU: {torch_directml.device_name(0)}"
except Exception as e:
    c.print(f"[yellow]DirectML unavailable ({e}) — falling back to CPU[/]")

from musicbot.generate import generate

bundle = ROOT / "models" / "maestro_bundle.pt"
if not bundle.exists():
    c.print(f"[red]missing {bundle} — copy it from WSL ~/music-bot/models/[/]")
    sys.exit(1)

out = ROOT / "output"
out.mkdir(exist_ok=True)
import time

precision = "fp32" if device != "cpu" else "int8"
c.print(f"backend: [cyan]{backend}[/] · precision: {precision}")
generate(
    str(bundle),
    str(out / f"opus_{time.strftime('%Y%m%d_%H%M%S')}.mid"),
    minutes=16.0,
    K=4,
    device=device,
    precision=precision,
    seed=int(sys.argv[1]) if len(sys.argv) > 1 else 0,
    log=c.print,
    console=c,
)
c.print("[green]done — MIDI is in the output folder[/]")
