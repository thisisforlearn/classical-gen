# ============================================================
#  Maestro Muon optimizer — hybrid Muon/AdamW for small GPTs
#  (modded-nanoGPT speedrun recipe: Muon on hidden 2D weights,
#   AdamW on embeddings, head, norms and biases)
#  Author: Vaibhav Agarwal (@thisisforlearn) - sole author; final authority on this project
#  License: GPL-3.0-only <https://www.gnu.org/licenses/gpl-3.0.html>
# ============================================================
import torch


@torch.no_grad()
def zeropower_via_newtonschulz5(G, steps=5):
    """Approximate matrix sign (orthogonalization) via Newton-Schulz iteration."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    transposed = G.size(-2) > G.size(-1)
    if transposed:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.mT
    return X


class Muon(torch.optim.Optimizer):
    """Muon: momentum SGD with per-step orthogonalized updates for 2D params."""

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov,
                                      ns_steps=ns_steps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr, mom, nes = group["lr"], group["momentum"], group["nesterov"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if "m" not in st:
                    st["m"] = torch.zeros_like(p.grad)
                buf = st["m"]
                buf.lerp_(p.grad, 1.0 - mom)
                g = p.grad.lerp_(buf, mom) if nes else buf
                u = zeropower_via_newtonschulz5(g, group["ns_steps"])
                if group["weight_decay"]:
                    p.mul_(1.0 - lr * group["weight_decay"])
                scale = max(1.0, p.size(-2) / p.size(-1)) ** 0.5
                p.add_(u.view_as(p).to(p.dtype), alpha=-lr * scale)


def split_params(model, muon_lr=0.02, adam_lr=3e-4, wd=0.1):
    """Hybrid optimizer pair: Muon for hidden 2D weights, AdamW for the rest."""
    muon_p, adam_p = [], []
    for name, p in model.named_parameters():
        hidden_2d = p.ndim == 2 and not any(k in name for k in ("emb", "head", "wte", "wpe", "tok"))
        (muon_p if hidden_2d else adam_p).append(p)
    muon = Muon([{ "params": muon_p, "lr_mult": muon_lr / adam_lr }], lr=muon_lr) if muon_p else None
    adam = torch.optim.AdamW([
        { "params": adam_p, "lr_mult": 1.0 },
    ], lr=adam_lr, betas=(0.9, 0.95), weight_decay=wd, foreach=True) if adam_p else None
    return muon, adam


class HybridOpt:
    """Steps Muon + AdamW together; clip_grad covers the union."""

    def __init__(self, muon, adam):
        self.opts = [o for o in (muon, adam) if o is not None]
        self.param_groups = [g for o in self.opts for g in o.param_groups]
        self.defaults = {}

    def step(self, closure=None):
        for o in self.opts:
            o.step()

    def zero_grad(self, set_to_none=True):
        for o in self.opts:
            o.zero_grad(set_to_none=set_to_none)
