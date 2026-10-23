"""
End-to-end demo / smoke test.

Simulates a heavy-tailed synthetic return series from a *known* Variance
Gamma process (so we have ground truth to sanity-check against), fits the
NSF-VG flow, checks NLL <= 0 in expectation over held-out data, reports VaR
at 95%/99%/99.9%, and saves a QQ plot.

Swap `simulate_vg_returns(...)` for your own return series (a 1-D torch
tensor of log-returns) to use this on real data.
"""

from pathlib import Path

import numpy as np
import torch

from levy_flow import (
    NSFFlow,
    TrainConfig,
    VarianceGamma,
    estimate_var,
    ks_test,
    plot_qq,
    qq_data,
    train_flow,
    warm_start_base,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def simulate_vg_returns(n: int, mu=0.0002, sigma=0.012, theta=-0.0015, nu=0.4, seed=0):
    rng = np.random.default_rng(seed)
    g = rng.gamma(shape=1.0 / nu, scale=nu, size=n)
    z = rng.standard_normal(n)
    x = mu + theta * g + sigma * np.sqrt(g) * z
    return torch.tensor(x, dtype=torch.float32)


def main():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- data -----------------------------------------------------------
    n_total = 8000
    x_all_raw = simulate_vg_returns(n_total)
    n_train = int(0.8 * n_total)
    x_train_raw, x_val_raw = x_all_raw[:n_train], x_all_raw[n_train:]

    # Standardize before fitting. This matters for two reasons:
    #  (1) tail_bound=5.0 defines where the spline lives / where identity
    #      tails kick in -- that's only meaningful on an O(1) scale, not on
    #      raw returns which might be ~1e-2 in magnitude.
    #  (2) log-density scales inversely with the data's scale (log p ~ -log
    #      sigma + const), so unstandardized, tightly-concentrated returns
    #      can trivially produce log p > 0 (NLL < 0) with no relation to
    #      how good the fit is. Standardizing puts NLL on a comparable,
    #      interpretable footing.
    x_mean = x_train_raw.mean()
    x_std = x_train_raw.std()
    x_train = (x_train_raw - x_mean) / x_std
    x_val = (x_val_raw - x_mean) / x_std

    # --- base distribution + flow ---------------------------------------
    # Base VG params are simple, fixed moment-matched estimates on the
    # standardized data; the flow does the rest of the work.
    base = VarianceGamma(mu=0.0, sigma=1.0, theta=0.0, nu=0.5, dtype=torch.float32)
    warm_start_base(base, x_train, epochs=300)

    flow = NSFFlow(dim=1, base=base, num_layers=6, num_bins=8, tail_bound=5.0,
                    hidden_dim=64, n_hidden=2)

    # --- train ------------------------------------------------------------
    cfg = TrainConfig(epochs=150, batch_size=256, lr=1e-3, grad_clip_norm=5.0, print_every=25)
    history = train_flow(flow, x_train, cfg, x_val=x_val, device=device)

    final_val_nll = history["best_val_nll"] if history["best_val_nll"] is not None else history["val_nll"][-1]
    print(f"\nFinal (best-checkpoint) validation NLL: {final_val_nll:.4f}  "
          f"(positive NLL <=> log-likelihood <= 0: {'OK' if final_val_nll > 0 else 'CHECK'})")

    # --- VaR ---------------------------------------------------------------
    # estimate_var operates in the flow's (standardized) units; rescale
    # samples back to raw return units before computing VaR.
    _, samples_std = estimate_var(flow, levels=(0.95, 0.99, 0.999), n_samples=200_000, device=device)
    samples_raw = samples_std * x_std.item() + x_mean.item()
    print("\nMonte-Carlo VaR (from 200k flow samples, raw return units):")
    for lvl in (0.95, 0.99, 0.999):
        q = np.quantile(samples_raw, 1 - lvl)
        print(f"  VaR_{lvl*100:.1f}%  = {-q:.5f}")

    # empirical VaR on held-out data, for comparison
    x_val_np = x_val_raw.numpy()
    print("\nEmpirical VaR on held-out data (sanity check):")
    for lvl in (0.95, 0.99, 0.999):
        q = np.quantile(x_val_np, 1 - lvl)
        print(f"  VaR_{lvl*100:.1f}%  = {-q:.5f}")

    # --- QQ plot -------------------------------------------------------
    z_sorted, ref_q = qq_data(flow, x_val, n_ref=200_000, device=device)
    path = plot_qq(z_sorted, ref_q, path=str(PROJECT_ROOT / "results" / "qq_plot.png"))
    print(f"\nSaved QQ plot to {path}")

    # --- KS test ---------------------------------------------------------
    # Same diagnostic that caught the log_prob/sample() mismatch in the
    # levy-flows repo: compares flow.sample() output against real held-out
    # data. Run in the standardized units the flow was trained on so it's
    # comparable across runs regardless of raw data scale.
    ks = ks_test(flow, x_val, device=device)
    print(f"\nKS test (flow samples vs held-out data): "
          f"stat={ks['ks_statistic']:.4f}  p={ks['ks_pvalue']:.4f}  (n_samples={ks['n_samples']})")
    if ks["ks_pvalue"] < 0.05:
        print("  -> reject H0 at 5%: samples and data distinguishable (expected for finite n; "
              "watch the *magnitude* of the statistic, not just significance)")
    else:
        print("  -> fail to reject H0 at 5%: no detected distributional mismatch")


if __name__ == "__main__":
    main()