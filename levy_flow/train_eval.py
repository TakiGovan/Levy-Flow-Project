"""
Training loop (NLL + Adam + grad clipping) and evaluation utilities
(VaR at multiple confidence levels, QQ plot) for the NSF-VG flow.
"""

import copy
from dataclasses import dataclass
from typing import Sequence
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .flow import NSFFlow


@dataclass
class TrainConfig:
    epochs: int = 200
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 0.0
    grad_clip_norm: float = 5.0
    print_every: int = 20
    lr_schedule: bool = True   # cosine anneal lr -> min_lr over training
    min_lr: float = 1e-5
    restore_best: bool = True  # reload the best-val-NLL checkpoint at the end


def train_flow(flow: NSFFlow, x_train: torch.Tensor, cfg: TrainConfig = TrainConfig(),
                x_val: torch.Tensor = None, device: str = "cpu"):
    """
    x_train: (N,) or (N, dim) tensor of training data.
    Minimizes NLL = -mean(log_prob(x)). Returns dict of loss history.

    If x_val is given and cfg.restore_best is True, the model's state at the
    epoch with the lowest val NLL is restored before returning -- guards
    against a run that oscillates near the end and finishes worse than its
    best point (which is common with a base distribution that's still being
    fit jointly with the flow).
    """
    flow = flow.to(device)
    if x_train.dim() == 1:
        x_train = x_train.unsqueeze(-1)
    x_train = x_train.to(device)

    loader = DataLoader(TensorDataset(x_train), batch_size=cfg.batch_size, shuffle=True)
    opt = torch.optim.Adam(flow.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = None
    if cfg.lr_schedule:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=cfg.epochs, eta_min=cfg.min_lr
        )

    best_val_nll = float("inf")
    best_state = None
    best_epoch = None

    history = {"train_nll": [], "val_nll": []}
    for epoch in range(1, cfg.epochs + 1):
        flow.train()
        epoch_nll = 0.0
        n_seen = 0
        for (xb,) in loader:
            opt.zero_grad()
            log_prob = flow.log_prob(xb)
            nll = -log_prob.mean()
            if not torch.isfinite(nll):
                continue  # skip a pathological batch rather than corrupt parameters
            nll.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(flow.parameters(), cfg.grad_clip_norm)
            if not torch.isfinite(gnorm):
                opt.zero_grad()
                continue  # defensive: clipping computed on a non-finite gradient is meaningless
            opt.step()
            epoch_nll += nll.item() * xb.shape[0]
            n_seen += xb.shape[0]
        train_nll = epoch_nll / n_seen if n_seen > 0 else float("nan")
        history["train_nll"].append(train_nll)

        val_nll = None
        if x_val is not None:
            flow.eval()
            with torch.no_grad():
                xv = x_val.unsqueeze(-1) if x_val.dim() == 1 else x_val
                val_nll = -flow.log_prob(xv.to(device)).mean().item()
            history["val_nll"].append(val_nll)

            if val_nll < best_val_nll:
                best_val_nll = val_nll
                best_epoch = epoch
                best_state = copy.deepcopy(flow.state_dict())

        if scheduler is not None:
            scheduler.step()

        if epoch % cfg.print_every == 0 or epoch == 1:
            msg = f"epoch {epoch:4d}/{cfg.epochs}  train NLL = {train_nll:8.4f}"
            if val_nll is not None:
                msg += f"   val NLL = {val_nll:8.4f}"
            if scheduler is not None:
                msg += f"   lr = {scheduler.get_last_lr()[0]:.2e}"
            print(msg)

    if cfg.restore_best and best_state is not None:
        flow.load_state_dict(best_state)
        print(f"\nRestored best checkpoint: epoch {best_epoch}, val NLL = {best_val_nll:.4f}")

    history["best_val_nll"] = best_val_nll if best_state is not None else None
    history["best_epoch"] = best_epoch
    return history


def warm_start_base(base, x_train: torch.Tensor, epochs: int = 300, lr: float = 1e-2,
                     grad_clip_norm: float = 1.0, device: str = "cpu"):
    """
    Fit the VG base distribution's own (mu, sigma, theta, nu) to x_train in
    isolation -- no flow involved -- before it's handed to NSFFlow for joint
    training. This gives the base a good starting point instead of letting
    joint optimization discover both the base and the flow from scratch
    together, which empirically can pull the base to a worse fit than it
    would reach on its own (verified: joint training landed nu=0.685 on one
    real run vs nu=1.2 for a clean fit of the base alone, with the clean fit's
    train NLL meaningfully better -- 1.2368 vs 1.2834).

    Gradient-clipped and NaN-guarded: an earlier unclipped version (lr=5e-2,
    no clipping) could oscillate mu/theta and drive nu on a sustained slide
    toward 0 -- a real degeneracy of VG-type likelihoods, since the density
    near x=mu is unbounded as nu->0 -- which eventually produced a NaN step
    a number of epochs into subsequent joint training. If a step's loss or
    gradient is ever non-finite here, that step is skipped rather than applied.
    """
    base = base.to(device)
    x = x_train.to(device)
    if x.dim() > 1:
        x = x.squeeze(-1)

    opt = torch.optim.Adam(base.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        nll = -base.log_prob(x).mean()
        if not torch.isfinite(nll):
            continue  # skip a pathological step rather than corrupt parameters
        nll.backward()
        torch.nn.utils.clip_grad_norm_(base.parameters(), grad_clip_norm)
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in base.parameters()):
            continue  # defensive: skip if clipping still leaves a non-finite gradient
        opt.step()

    final_nll = -base.log_prob(x).mean().item()
    return final_nll


@torch.no_grad()
def estimate_var(flow: NSFFlow, levels: Sequence[float] = (0.95, 0.99, 0.999),
                  n_samples: int = 200_000, device: str = "cpu"):
    """
    Monte-Carlo VaR from the fitted flow. Convention: x is a P&L / return series,
    losses are negative returns. VaR_alpha is the (positive) loss threshold
    exceeded with probability (1 - alpha):

        VaR_alpha = -quantile_{1-alpha}(returns)

    Returns dict {level: VaR}.
    """
    flow.eval()
    samples = flow.sample(n_samples).squeeze(-1).cpu().numpy()
    out = {}
    for alpha in levels:
        q = np.quantile(samples, 1 - alpha)
        out[alpha] = -q
    return out, samples


@torch.no_grad()
def qq_data(flow: NSFFlow, x_data: torch.Tensor, n_ref: int = 200_000, device: str = "cpu"):
    """
    Base-space QQ diagnostic: push real data through the fitted flow to base
    space, z = T(x). If the fit is good, z should look like i.i.d. draws from
    the VG base distribution. We compare sorted z against sorted quantiles of
    a large reference sample drawn directly from the base VarianceGamma.

    Returns (z_sorted, ref_quantiles) both length == len(x_data), ready to
    scatter-plot against each other.
    """
    flow.eval()
    x = x_data.unsqueeze(-1) if x_data.dim() == 1 else x_data
    z, _ = flow.forward(x.to(device))
    z = z.squeeze(-1).cpu().numpy()
    z_sorted = np.sort(z)

    ref = flow.base.rsample((n_ref,)).cpu().numpy()
    ref_sorted = np.sort(ref)

    probs = (np.arange(1, len(z_sorted) + 1) - 0.5) / len(z_sorted)
    ref_quantiles = np.quantile(ref_sorted, probs)
    return z_sorted, ref_quantiles


@torch.no_grad()
def ks_test(flow: NSFFlow, x_data: torch.Tensor, n_samples: int = None, device: str = "cpu"):
    """
    Two-sample Kolmogorov-Smirnov test between flow.sample() and real data,
    in the same units as x_data (pass raw or standardized consistently with
    what x_data is in).

    This is the diagnostic that catches a log_prob/sample() inconsistency:
    a low training loss only reflects log_prob, but the KS test compares
    what sample() actually produces against the real distribution -- if
    those two don't describe the same law, KS will catch it even when the
    loss looks good.

    n_samples defaults to max(10_000, 10 * len(x_data)), following the same
    convention used elsewhere in this codebase for sample-based diagnostics.
    """
    from scipy.stats import ks_2samp

    flow.eval()
    x_np = x_data.detach().cpu().numpy().flatten()
    if n_samples is None:
        n_samples = max(10_000, 10 * len(x_np))

    samples = flow.sample(n_samples)
    samples_np = samples.detach().cpu().numpy().flatten()

    stat, pvalue = ks_2samp(samples_np, x_np)
    return {"ks_statistic": float(stat), "ks_pvalue": float(pvalue), "n_samples": n_samples}


def plot_qq(z_sorted: np.ndarray, ref_quantiles: np.ndarray, path: str = "qq_plot.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(ref_quantiles, z_sorted, s=8, alpha=0.5, label="data (base space) vs VG reference")
    lo = min(ref_quantiles.min(), z_sorted.min())
    hi = max(ref_quantiles.max(), z_sorted.max())
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="y = x")
    ax.set_xlabel("Theoretical VG quantiles")
    ax.set_ylabel("Empirical quantiles (flow-transformed data)")
    ax.set_title("QQ plot: flow-transformed data vs Variance-Gamma base")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path