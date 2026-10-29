# ============================================================
#  Maestro trainer — optimized CPU/GPU pretraining loop
#  Author: thisisforlearn (sole author; final authority on this project)
#  License: GPL-3.0-only <https://www.gnu.org/licenses/gpl-3.0.html>
# ============================================================
from __future__ import annotations
import math, time, os
import numpy as np
import torch

try:
    from musicbot.model import GPT, GEN_CFG
except ImportError:
    from model import GPT, GEN_CFG
try:
    from musicbot.dataset import PackedData
except ImportError:
    from dataset import PackedData


def cosine_lr(step, total, warmup, base):
    if step < warmup:
        return base * step / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return base * 0.5 * (1 + math.cos(math.pi * min(1.0, p)))


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = [p.detach().clone().float() for p in model.parameters()]

    @torch.no_grad()
    def update(self, model):
        for s, p in zip(self.shadow, model.parameters()):
            s.mul_(self.decay).add_(p.detach().float(), alpha=1 - self.decay)

    @torch.no_grad()
    def copy_to(self, model):
        for s, p in zip(self.shadow, model.parameters()):
            p.copy_(s.to(p.dtype))

    def state(self):
        return [s.clone() for s in self.shadow]

    def load(self, state):
        self.shadow = [s.clone() for s in state]


def pretrain(model, data, steps, bs, lr, device, bf16=True, log=print, ckpt_fn=None, ckpt_every=500,
             amp_device="cpu", parallel=False, accum=1, max_seconds=None, use_bar=False, ls=0.0):
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1, foreach=True)
    net = model
    if parallel and device == "cuda" and torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
        log(f"DataParallel over {torch.cuda.device_count()} GPUs")
    ema = EMA(model)
    rng = np.random.default_rng(1337)
    warm = max(50, steps // 20)
    t0, tok_acc, loss_ema = time.time(), 0, None
    t_start = time.time()
    bar = ptask = None
    if use_bar:
        from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
        bar = Progress(SpinnerColumn(), TextColumn("[bold cyan]pretrain "),
                       BarColumn(bar_width=44), TextColumn("[green]{task.fields[m]}"),
                       TimeElapsedColumn(), TimeRemainingColumn())
        bar.start()
        ptask = bar.add_task("pretrain", total=steps, m="")
    for step in range(1, steps + 1):
        lr_s = cosine_lr(step, steps, warm, lr)
        for g_ in opt.param_groups:
            g_["lr"] = lr_s
        for _ in range(accum):
            x, y = data.sample_batch(bs, rng)
            x = torch.from_numpy(x).to(device, non_blocking=True)
            y = torch.from_numpy(y).to(device, non_blocking=True)
            with torch.autocast(device_type=amp_device, dtype=torch.bfloat16, enabled=bf16):
                _, loss = net(x, y, ls=ls)
            if loss.dim() > 0:
                loss = loss.mean()
            loss.backward()
            tok_acc += x.numel()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        ema.update(model)
        l = loss.item()
        loss_ema = l if loss_ema is None else 0.95 * loss_ema + 0.05 * l
        if bar is not None:
            bar.update(ptask, completed=step,
                       m=f"loss {loss_ema:.3f} · {tok_acc / max(1e-9, time.time() - t0) / 1e3:.1f}k tok/s")
        elif step % 25 == 0 or step == 1:
            dt = time.time() - t0
            log(f"step {step}/{steps}  loss {loss_ema:.3f}  lr {lr_s:.2e}  {tok_acc / dt / 1e3:.1f}k tok/s")
            t0, tok_acc = time.time(), 0
        if step % 25 == 0:
            t0, tok_acc = time.time(), 0
        if ckpt_fn and step % ckpt_every == 0:
            ckpt_fn(model, step, ema)
        if max_seconds and (time.time() - t_start) > max_seconds:
            log(f"time budget reached at step {step} — stopping")
            if ckpt_fn:
                ckpt_fn(model, step, ema)
            break
    if bar is not None:
        bar.stop()
    return model, ema
