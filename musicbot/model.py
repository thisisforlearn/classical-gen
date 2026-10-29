# ============================================================
#  Maestro models — GPT generator + scalar critic (RoPE/RMSNorm/SwiGLU)
#  Author: thisisforlearn (sole author; final authority on this project)
#  License: GPL-3.0-only <https://www.gnu.org/licenses/gpl-3.0.html>
# ============================================================
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from musicbot.tokenizer import VOCAB_SIZE
except ImportError:
    from tokenizer import VOCAB_SIZE

from torch.nn.attention import SDPBackend, sdpa_kernel


class Config(dict):
    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            raise AttributeError(k)


GEN_CFG = Config(vocab=VOCAB_SIZE, d=512, n_layer=10, n_head=8, ctx=1024, ff=1360, drop=0.0)
CRIT_CFG = Config(vocab=VOCAB_SIZE, d=384, n_layer=6, n_head=6, ctx=512, ff=1024, drop=0.0)


_ROPE_CACHE = {}


def _rope_cache(seq_len, head_dim, device, dtype):
    key = (seq_len, head_dim, str(device), str(dtype))
    r = _ROPE_CACHE.get(key)
    if r is None:
        inv = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        f = torch.outer(t, inv)
        r = (torch.cos(f).to(dtype), torch.sin(f).to(dtype))
        if len(_ROPE_CACHE) > 64:
            _ROPE_CACHE.clear()
        _ROPE_CACHE[key] = r
    return r


def _apply_rope(x, cos, sin):
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)


class Attn(nn.Module):
    def __init__(self, d, n_head):
        super().__init__()
        assert d % n_head == 0, f"d={d} must be divisible by n_head={n_head}"
        self.n_head = n_head
        self.hd = d // n_head
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)

    def forward(self, x, cos, sin, cache=None):
        B, T, D = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_head, self.hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)
        if cache is not None:
            idx = cache["idx"]
            cache["k"][:, :, idx : idx + T] = k
            cache["v"][:, :, idx : idx + T] = v
            k = cache["k"][:, :, : idx + T]
            v = cache["v"][:, :, : idx + T]
            cache["idx"] = idx + T
            if idx == 0:
                y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            elif T == 1:
                if x.is_cuda:
                    y = F.scaled_dot_product_attention(q, k, v)
                else:
                    with sdpa_kernel(SDPBackend.MATH):
                        y = F.scaled_dot_product_attention(q, k, v)
            else:
                L = idx + T
                pos_k = torch.arange(L, device=x.device)
                pos_q = torch.arange(T, device=x.device) + idx
                mask = (pos_k[None, :] <= pos_q[:, None]).bool()
                y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        else:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=T > 1)
        y = y.transpose(1, 2).reshape(B, T, D)
        return self.proj(y)


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n1 = nn.RMSNorm(cfg.d)
        self.attn = Attn(cfg.d, cfg.n_head)
        self.n2 = nn.RMSNorm(cfg.d)
        self.up = nn.Linear(cfg.d, cfg.ff, bias=False)
        self.gate = nn.Linear(cfg.d, cfg.ff, bias=False)
        self.down = nn.Linear(cfg.ff, cfg.d, bias=False)

    def forward(self, x, cos, sin, cache=None):
        x = x + self.attn(self.n1(x), cos, sin, cache)
        x = x + self.down(F.silu(self.gate(self.n2(x))) * self.up(self.n2(x)))
        return x


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        cfg = Config(cfg)
        self.cfg = cfg
        self.emb = nn.Embedding(cfg.vocab, cfg.d)
        self.drop = nn.Dropout(cfg.drop)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm_f = nn.RMSNorm(cfg.d)
        self.head = nn.Linear(cfg.d, cfg.vocab, bias=False)
        self.apply(self._init)
        for b in self.blocks:
            nn.init.normal_(b.down.weight, std=0.02 / math.sqrt(2 * cfg.n_layer))
            nn.init.normal_(b.attn.proj.weight, std=0.02 / math.sqrt(2 * cfg.n_layer))

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, targets=None, cache=None, pos_offset=0, ls=0.0):
        B, T = idx.shape
        x = self.drop(self.emb(idx))
        need = pos_offset + T
        size = self.cfg.ctx if need <= self.cfg.ctx else need
        cos, sin = _rope_cache(size, self.cfg.d // self.cfg.n_head, x.device, x.dtype)
        if pos_offset or need != size:
            cos, sin = cos[pos_offset:need], sin[pos_offset:need]
        for i, b in enumerate(self.blocks):
            x = b(x, cos, sin, cache[i] if cache is not None else None)
        x = self.norm_f(x)
        if targets is not None:
            logits = self.head(x)
            return logits, F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1),
                                           ignore_index=-100, label_smoothing=ls)
        return self.head(x[:, [-1]] if cache is not None else x), None

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


class Critic(nn.Module):
    def __init__(self, cfg=CRIT_CFG):
        super().__init__()
        cfg = Config(cfg)
        self.cfg = cfg
        self.emb = nn.Embedding(cfg.vocab, cfg.d)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm_f = nn.RMSNorm(cfg.d)
        self.head = nn.Linear(cfg.d, 1, bias=False)
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx):
        x = self.emb(idx)
        T = idx.shape[1]
        cos, sin = _rope_cache(T, self.cfg.d // self.cfg.n_head, x.device, x.dtype)
        for b in self.blocks:
            x = b(x, cos, sin)
        x = self.norm_f(x).mean(dim=1)
        return self.head(x).squeeze(-1)
