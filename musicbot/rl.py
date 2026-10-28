# ============================================================
#  Maestro RL — theory rewards, critic training, REINFORCE fine-tuning
#  Author: Vaibhav (sole author; final authority on this project)
#  License: GPL-3.0-only <https://www.gnu.org/licenses/gpl-3.0.html>
# ============================================================
from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F

try:
    from musicbot.tokenizer import (
        BAR, EOS, SEP, BOS, POS0, TEMPO0, TS0, PITCH0, DUR0, VEL0, FORM0, VOCAB_SIZE, PITCH_LO,
    )
except ImportError:
    from tokenizer import (
        BAR, EOS, SEP, BOS, POS0, TEMPO0, TS0, PITCH0, DUR0, VEL0, FORM0, VOCAB_SIZE, PITCH_LO,
    )

MAJOR = {0, 2, 4, 5, 7, 9, 11}
TONICISH = {0, 4, 7, 9}


def theory_reward(tokens) -> float:
    t = np.asarray(tokens, dtype=np.int64)
    pitches = t[(t >= PITCH0) & (t < DUR0)] - PITCH0 + PITCH_LO
    if len(pitches) < 16:
        return 0.0
    bars = max(1, int((t == BAR).sum()))
    pc = np.bincount(pitches % 12, minlength=12).astype(np.float64)
    p = pc / pc.sum()
    ent = -(p[p > 0] * np.log(p[p > 0])).sum() / np.log(12)
    key = int(pc.argmax())
    in_key = float((np.isin((pitches - key) % 12, list(MAJOR))).mean())
    density = len(pitches) / bars
    dens_ok = float(np.exp(-((density - 14.0) ** 2) / 200.0))
    leaps = np.abs(np.diff(pitches.astype(np.int16)))
    smooth = float((leaps <= 12).mean())
    tonic_end = float((((pitches[-8:] - key) % 12)[:, None] == list(TONICISH)).any())
    barset = {}
    cur = []
    for tok in t:
        if tok == BAR:
            if cur:
                barset.setdefault(hash(tuple(cur)) % 997, 0)
                barset[hash(tuple(cur)) % 997] += 1
            cur = []
        elif PITCH0 <= tok < DUR0:
            cur.append(int(tok))
    rep = max(barset.values()) / max(1, len(barset)) if barset else 0.0
    r = 0.9 * in_key + 0.8 * (1.0 - ent) + 0.7 * dens_ok + 0.6 * smooth + 0.5 * tonic_end + 0.4 * rep
    return float(r / 3.9)


@torch.no_grad()
def sample_continuations(model, prompts, max_new, temp=1.0, topk=40, topp=0.95, rep_pen=1.15, device="cpu"):
    model.eval()
    B, T = prompts.shape
    prompts = prompts.to(device)
    idx = prompts.clone()
    nl, H, hd = model.cfg.n_layer, model.cfg.n_head, model.cfg.d // model.cfg.n_head
    caches = [{"k": torch.zeros(B, H, T + max_new, hd, device=device),
               "v": torch.zeros(B, H, T + max_new, hd, device=device), "idx": 0} for _ in range(nl)]
    on_cuda = str(device).startswith("cuda")
    amp = torch.autocast(device_type="cuda" if on_cuda else "cpu", dtype=torch.bfloat16, enabled=on_cuda)
    with amp:
        logits, _ = model(prompts[:, -model.cfg.ctx:], cache=caches)
    cur = T
    for _ in range(max_new):
        logits = logits[:, -1, :].float()
        for b in range(B):
            row = idx[b, -256:]
            uniq, cnt = torch.unique(row, return_counts=True)
            logits[b, uniq] /= rep_pen ** cnt.clamp(max=6).float()
        logits /= max(0.05, temp)
        if topk:
            kth = torch.topk(logits, min(topk, logits.size(-1)), dim=-1)
            logits[logits < kth.values[:, [-1]]] = -float("inf")
        probs = torch.softmax(logits, dim=-1)
        if not torch.isfinite(probs).all():
            probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
            s = probs.sum(dim=-1, keepdim=True)
            probs = torch.where(s > 0, probs / s, torch.full_like(probs, 1.0 / probs.size(-1)))
        if topp:
            sp, si = torch.sort(probs, descending=True, dim=-1)
            cum = torch.cumsum(sp, dim=-1)
            mask = cum - sp > topp
            sp[mask] = 0
            sp /= sp.sum(dim=-1, keepdim=True)
            nxt = si.gather(-1, torch.multinomial(sp, 1)).squeeze(-1)
        else:
            nxt = torch.multinomial(probs, 1).squeeze(-1)
        nxt = nxt.clamp(0, model.cfg.vocab - 1)
        idx = torch.cat([idx, nxt[:, None]], dim=1)
        cur += 1
        if (nxt == EOS).all():
            break
        with amp:
            logits, _ = model(nxt[:, None], cache=caches, pos_offset=cur - 1)
    return idx[:, T:]


def train_critic(critic, gen_model, data, steps, bs, lr, device, log=print, G=4, seg=192, use_bar=False):
    critic.to(device).train()
    gen_model.to(device).eval()
    opt = torch.optim.AdamW(critic.parameters(), lr=lr, weight_decay=0.01)
    rng = np.random.default_rng(7)
    ema = None
    bar = ptask = None
    if use_bar:
        from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
        bar = Progress(SpinnerColumn(), TextColumn("[bold magenta]critic   "),
                       BarColumn(bar_width=44), TextColumn("[green]{task.fields[m]}"),
                       TimeElapsedColumn(), TimeRemainingColumn())
        bar.start()
        ptask = bar.add_task("critic", total=steps, m="")
    for step in range(1, steps + 1):
        xr, yr = data.sample_batch(bs, rng)
        seg_total = 64 + seg
        real = torch.from_numpy(xr).to(device)[:, -seg_total:]
        with torch.no_grad():
            seeds = torch.from_numpy(data.sample_batch(1, rng)[0][:, :64]).to(device).repeat(G, 1)
            fake = torch.cat([seeds, sample_continuations(gen_model, seeds, seg, device=device)], dim=1)
        real_lab = torch.ones(bs, device=device) * 0.9
        fake_lab = torch.zeros(fake.size(0), device=device) + 0.05
        x = torch.cat([real, fake], 0)
        lab = torch.cat([real_lab, fake_lab], 0)
        logit = critic(x[:, -critic.cfg.ctx :])
        loss = F.binary_cross_entropy_with_logits(logit, lab)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
        opt.step()
        ema = loss.item() if ema is None else 0.95 * ema + 0.05 * loss.item()
        with torch.no_grad():
            acc = (((logit > 0).float() == lab).float().mean().item())
        if bar is not None:
            bar.update(ptask, completed=step, m=f"bce {ema:.3f} · acc {acc:.2f}")
        elif step % 25 == 0 or step == 1:
            log(f"critic step {step}/{steps}  bce {ema:.3f}  acc {acc:.2f}")
    if bar is not None:
        bar.stop()
    return critic


def reinforce(gen_model, ref_model, critic, data, steps, device, log=print, G=8, seg=256, lr=1e-5, beta=0.05,
              use_bar=False):
    gen_model.to(device).train()
    ref_model.to(device).eval()
    critic.to(device).eval()
    opt = torch.optim.AdamW(gen_model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.01)
    rng = np.random.default_rng(11)
    bar = ptask = None
    if use_bar:
        from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
        bar = Progress(SpinnerColumn(), TextColumn("[bold yellow]rl       "),
                       BarColumn(bar_width=44), TextColumn("[green]{task.fields[m]}"),
                       TimeElapsedColumn(), TimeRemainingColumn())
        bar.start()
        ptask = bar.add_task("rl", total=steps, m="")
    for step in range(1, steps + 1):
        x, _ = data.sample_batch(1, rng)
        prompt = torch.from_numpy(x[:, :256]).to(device).repeat(G, 1)
        cont = sample_continuations(gen_model, prompt, seg, temp=1.0, device=device)
        full = torch.cat([prompt, cont], 1)[:, -(gen_model.cfg.ctx) :]
        with torch.no_grad():
            r_critic = torch.sigmoid(critic(full[:, -critic.cfg.ctx :]) / 4.0)
        r_theory = torch.tensor([theory_reward(row.tolist()) for row in cont], device=device)
        r_base = r_critic + 0.5 * r_theory
        adv = (r_base - r_base.mean()) / (r_base.std() + 1e-4)
        logits_ref, _ = ref_model(full[:, :-1])
        lp_ref = -F.cross_entropy(logits_ref.reshape(-1, logits_ref.size(-1)), full[:, 1:].reshape(-1), reduction="none").view(G, -1)
        logits, _ = gen_model(full[:, :-1])
        lp = -F.cross_entropy(logits.reshape(-1, logits.size(-1)), full[:, 1:].reshape(-1), reduction="none").view(G, -1)
        shaped = adv[:, None] - beta * (lp - lp_ref).detach() / seg
        loss = -(shaped * lp).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(gen_model.parameters(), 1.0)
        opt.step()
        msg = (f"loss {loss.item():.3f} · critic {r_critic.mean().item():.3f}"
               f" · theory {r_theory.mean().item():.3f}")
        if bar is not None:
            bar.update(ptask, completed=step, m=msg)
        elif step % 10 == 0 or step == 1:
            log(f"rl step {step}/{steps}  {msg}")
    if bar is not None:
        bar.stop()
    return gen_model
