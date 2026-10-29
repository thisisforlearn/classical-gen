# ============================================================
#  Maestro generator — hierarchical 16-min composition, best-of-N, AVX2 CPU
#  Author: Vaibhav Agarwal (@thisisforlearn) - sole author; final authority on this project
#  License: GPL-3.0-only <https://www.gnu.org/licenses/gpl-3.0.html>
# ============================================================
from __future__ import annotations
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from rich.progress import (
    Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn,
)

try:
    from musicbot.model import GPT, Critic
except ImportError:
    from model import GPT, Critic
try:
    from musicbot.tokenizer import (
        BOS, EOS, SEP, BAR, TEMPO0, TS0, FORM0, FORMS, decode_tokens, duration_sec_weighted,
    )
except ImportError:
    from tokenizer import (
        BOS, EOS, SEP, BAR, TEMPO0, TS0, FORM0, FORMS, decode_tokens, duration_sec_weighted,
    )
try:
    from musicbot.rl import theory_reward
except ImportError:
    from rl import theory_reward

import os


def _is_cpu(device):
    return isinstance(device, str) and str(device).startswith("cpu")

FORM_PLANS = [
    ["intro", "A", "A", "B", "A", "C", "A", "coda"],
    ["intro", "A", "B", "A2", "dev", "recap", "coda"],
    ["intro", "A", "B", "B", "A", "C", "A2", "coda"],
    ["intro", "A", "A2", "B", "A", "dev", "recap", "C", "coda"],
]


def load_bundle(path, device="cpu", precision="int8", log=print):
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    gcfg, ccfg = bundle["gen_cfg"], bundle["crit_cfg"]
    gen = GPT(gcfg)
    gen.load_state_dict(bundle["gen"])
    crit = Critic(ccfg)
    crit.load_state_dict(bundle["critic"])
    if _is_cpu(device):
        torch.set_num_threads(min(torch.get_num_threads(), os.cpu_count() or 1))
        if precision == "int8":
            from torch.ao.quantization import quantize_dynamic

            gen = quantize_dynamic(gen.eval(), {nn.Linear}, dtype=torch.qint8)
            log("generator: int8 dynamic quantization (oneDNN/AVX2 path)")
        elif precision == "bf16":
            log("generator: bfloat16 autocast (oneDNN path)")
        else:
            log("generator: fp32")
    gen = gen.to(device).eval()
    crit = crit.to(device).eval()
    return gen, crit, bundle


@torch.no_grad()
def _gen_batch(gen, prompts, max_new, temp, topk, topp, rep_pen, device, autocast):
    B, T = prompts.shape
    prompts = prompts.to(device)
    nl = gen.cfg.n_layer
    H, hd = gen.cfg.n_head, gen.cfg.d // gen.cfg.n_head
    caches = [{"k": torch.zeros(B, H, T + max_new, hd, device=device),
               "v": torch.zeros(B, H, T + max_new, hd, device=device), "idx": 0} for _ in range(nl)]
    ctx = gen.cfg.ctx
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=autocast):
        logits, _ = gen(prompts[:, -ctx:], cache=caches)
    idx = prompts.clone()
    cur = T
    for _ in range(max_new):
        logits = logits[:, -1, :].float()
        for b in range(B):
            row = idx[b, -256:]
            uniq, cnt = torch.unique(row, return_counts=True)
            logits[b, uniq] /= rep_pen ** cnt.clamp(max=6).float()
        logits /= max(0.05, temp)
        if topk:
            tv, _ = torch.topk(logits, min(topk, logits.size(-1)), dim=-1)
            logits[logits < tv[:, [-1]]] = -float("inf")
        probs = torch.softmax(logits, dim=-1)
        if topp:
            sp, si = torch.sort(probs, descending=True, dim=-1)
            cum = torch.cumsum(sp, dim=-1)
            sp[cum - sp > topp] = 0
            sp /= sp.sum(dim=-1, keepdim=True)
            nxt = si.gather(-1, torch.multinomial(sp, 1)).squeeze(-1)
        else:
            nxt = torch.multinomial(probs, 1).squeeze(-1)
        idx = torch.cat([idx, nxt[:, None]], dim=1)
        cur += 1
        if (nxt == EOS).all():
            break
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=autocast):
            logits, _ = gen(nxt[:, None], cache=caches, pos_offset=cur - 1)
    return idx[:, T:]


def _cut_section(toks):
    t = np.asarray(toks)
    for stop in (EOS, SEP):
        w = np.where(t == stop)[0]
        if len(w):
            t = t[: w[0]]
    bars = np.where(t == BAR)[0]
    if len(bars) > 1:
        t = t[: bars[-1]]
    return t


def compose(gen, crit, minutes=16.0, K=4, temp=1.0, topk=20, topp=0.95, device="cpu",
            precision="int8", seed=0, log=print, console=None):
    import random
    torch.manual_seed(seed)
    rng = random.Random(seed * 2654435761 % (2 ** 31))
    temp = max(0.9, temp * (0.96 + 0.08 * rng.random()))
    rep = 1.15 + 0.1 * rng.random()
    autocast = _is_cpu(device) and precision == "bf16"
    target = minutes * 60.0 + 30.0
    plan = list(FORM_PLANS[rng.randrange(len(FORM_PLANS))])
    if rng.random() < 0.7:
        plan.insert(rng.randrange(2, len(plan) - 1), rng.choice(["A2", "B", "C", "dev"]))
    while len(plan) < 24:
        plan += ["A", "B", "C", "A2"][seed % 4 :]
    plan = plan[:24]
    all_toks = []
    sec_len, tail_len = 512, 384
    made = 0.0
    pbar = Progress(
        SpinnerColumn(), TextColumn("[bold cyan]{task.description}"), BarColumn(bar_width=40),
        TextColumn("[green]{task.fields[done]}"), TimeElapsedColumn(), TimeRemainingColumn(),
        console=console,
    )
    tok_total, t_start = 0, time.time()
    tgt_label = f"{int(minutes)}:{int(round((minutes % 1) * 60)):02d}"
    with pbar:
        main = pbar.add_task("composing", total=int(target), done=f"0:00 / {tgt_label}")
        for si, form in enumerate(plan):
            if made >= target:
                break
            form_id = FORM0 + FORMS.index(form)
            prompt = [BOS, form_id] + ([t for t in all_toks[-tail_len:]] or [])
            pb = torch.tensor(prompt, dtype=torch.long)[None, :].repeat(K, 1)
            psec = pbar.add_task(f"section {si + 1} [{form}]", total=sec_len, done="")
            with torch.no_grad():
                cont = _gen_batch(gen, pb, sec_len, temp, topk, topp, rep, device, autocast)
            pbar.remove_task(psec)
            best, best_s = None, -1e9
            for k in range(K):
                seg = _cut_section(cont[k].tolist())
                if len(seg) < 48:
                    continue
                th = theory_reward(seg)
                if crit is not None:
                    full = torch.tensor((prompt + seg.tolist())[-512:], dtype=torch.long)[None].to(device)
                    with torch.no_grad():
                        sc = torch.sigmoid(crit(full)).item()
                    s = sc + 0.5 * th
                else:
                    s = th
                if s > best_s:
                    best, best_s = seg, s
            if best is None:
                continue
            all_toks.extend(best.tolist())
            tok_total += len(best)
            made = duration_sec_weighted(all_toks)
            rate = tok_total / max(1e-9, time.time() - t_start)
            pbar.update(main, completed=int(min(made, target)), done=f"{int(made // 60)}:{int(made % 60):02d} / {tgt_label}  {rate:.0f} tok/s  score {best_s:.2f}")
    return all_toks


def generate(bundle_path, out_path, minutes=16.0, K=4, device="cpu", precision="int8",
             seed=0, log=print, console=None):
    gen, crit, _ = load_bundle(bundle_path, device, precision, log)
    log(f"composing {minutes:.0f} minutes, best-of-{K} per section...")
    toks = compose(gen, crit, minutes, K, device=device, precision=precision, seed=seed,
                   log=log, console=console)
    midi = decode_tokens(toks)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    midi.save(out_path)
    log(f"saved {out_path}  ({len(toks)} tokens, {duration_sec_weighted(toks) / 60:.1f} min)")
    return out_path
