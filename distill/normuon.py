"""NorMuon with Polar Express orthogonalization, for LoRA matrices.

Muon (Jordan et al. 2024) replaces each 2D update with the orthogonal factor of its
momentum. Polar Express (Amsel et al. 2025) computes that factor with per-iteration
optimal polynomial coefficients instead of Muon's fixed Newton-Schulz ones. NorMuon
(Li et al. 2025) then evens out the scale across neurons (rows) with an Adam-style
second moment, keeping the update's overall norm.

The update is scaled to an RMS of 0.2 (Moonshot's rule), so learning rates mean about
what they do for AdamW. Non-2D parameters (none in a PEFT LoRA) fall back to AdamW.
"""

from __future__ import annotations

import torch

# Polar Express coefficients (a, b, c) for X <- aX + b(XX^T)X + c(XX^T)^2 X, 5 steps,
# as used in modded-nanogpt.
POLAR_EXPRESS = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]


@torch.no_grad()
def polar_express(g: torch.Tensor) -> torch.Tensor:
    """Approximate orthogonal factor U V^T of a 2D matrix."""
    x = g.float()
    tall = x.size(0) > x.size(1)
    if tall:
        x = x.T
    x = x / (x.norm() * 1.02 + 1e-6)
    for a, b, c in POLAR_EXPRESS:
        s = x @ x.T
        x = a * x + (b * s + c * (s @ s)) @ x
    return x.T if tall else x


class PolarNorMuon(torch.optim.Optimizer):
    def __init__(self, params, lr=2e-4, momentum=0.95, beta2=0.95, weight_decay=0.0,
                 adam_betas=(0.9, 0.999), eps=1e-8):
        super().__init__(params, dict(lr=lr, momentum=momentum, beta2=beta2, weight_decay=weight_decay,
                                      adam_betas=adam_betas, eps=eps))

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr, wd = group["lr"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if wd:
                    p.mul_(1 - lr * wd)
                if p.ndim != 2:
                    self._adam(p, state, group)
                    continue
                if not state:
                    state["m"] = torch.zeros_like(p, dtype=torch.float32)
                    state["v"] = torch.zeros(p.size(0), 1, device=p.device, dtype=torch.float32)
                m = state["m"]
                m.lerp_(p.grad.float(), 1 - group["momentum"])
                # Nesterov: orthogonalize the look-ahead gradient.
                o = polar_express(p.grad.float().lerp(m, group["momentum"]))
                # NorMuon: per-row second moment of the orthogonal update, then restore its norm.
                row_sq = o.square().mean(dim=1, keepdim=True)
                state["v"].lerp_(row_sq, 1 - group["beta2"])
                before = o.norm()
                o = o / (state["v"].sqrt() + group["eps"])
                o = o * (before / o.norm().clamp_min(1e-10))
                # RMS 0.2 update, like AdamW's typical RMS.
                o = o * (0.2 * max(p.shape) ** 0.5)
                p.add_(o.to(p.dtype), alpha=-lr)
        return loss

    @staticmethod
    def _adam(p, state, group):
        b1, b2 = group["adam_betas"]
        if not state:
            state["step"], state["m"], state["v"] = 0, torch.zeros_like(p), torch.zeros_like(p)
        state["step"] += 1
        state["m"].lerp_(p.grad, 1 - b1)
        state["v"].lerp_(p.grad.square(), 1 - b2)
        m_hat = state["m"] / (1 - b1 ** state["step"])
        v_hat = state["v"] / (1 - b2 ** state["step"])
        p.addcdiv_(m_hat, v_hat.sqrt() + group["eps"], value=-group["lr"])
