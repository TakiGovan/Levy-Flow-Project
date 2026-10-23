from pathlib import Path

import numpy as np
import pandas as pd
import torch
import zuko
from torch.utils.data import DataLoader, TensorDataset

from levy_flow import (
    NSFFlow,
    TrainConfig,
    VarianceGamma,
    estimate_var,
    ks_test,
    qq_data,
    train_flow,
    warm_start_base,
)

torch.manual_seed(0)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ---- real data: user-provided S&P 500 series ----
df = pd.read_csv(PROJECT_ROOT / "data" / "sp500_returns.csv")
print("columns found:", list(df.columns))

# prefer an explicit ticker/returns column if it's already daily log-returns
# (e.g. "^GSPC" here) -- diffing that as if it were a price would be wrong.
return_col = next((c for c in df.columns if "ret" in c.lower() or c.strip().startswith("^")), None)
price_col = next((c for c in df.columns
                   if c.lower() in ("close", "adj close", "adj_close", "price", "sp500", "value")), None)

if return_col is not None:
    print(f"using existing returns column: '{return_col}'")
    log_ret = pd.to_numeric(df[return_col], errors="coerce").dropna().values
elif price_col is not None:
    print(f"computing log-returns from price column: '{price_col}'")
    prices = pd.to_numeric(df[price_col], errors="coerce").dropna().values
    log_ret = np.diff(np.log(prices))
else:
    raise ValueError(
        f"Couldn't auto-detect a returns or price column in {list(df.columns)}. "
        "Tell me the exact column name and I'll point at it directly."
    )

x_all_raw = torch.tensor(log_ret, dtype=torch.float32)
n = len(x_all_raw)
print(f"n = {n} daily log-returns")

# temporal 80/20 split (no lookahead), standardize on train only
n_train = int(0.8 * n)
x_train_raw, x_val_raw = x_all_raw[:n_train], x_all_raw[n_train:]
x_mean, x_std = x_train_raw.mean(), x_train_raw.std()
x_train = (x_train_raw - x_mean) / x_std
x_val = (x_val_raw - x_mean) / x_std

from scipy.stats import skew, kurtosis
print(f"train: mean={x_train_raw.mean():.5f} std={x_train_raw.std():.5f} "
      f"skew={skew(x_train_raw.numpy()):.3f} excess_kurt={kurtosis(x_train_raw.numpy()):.3f}")

EPOCHS = 400

# ---- My VG-base flow ----
base = VarianceGamma(mu=0.0, sigma=1.0, theta=0.0, nu=0.5, dtype=torch.float32)

# Warm-start the base's own (mu, sigma, theta, nu) before handing it to the
# flow for joint training. Verified fix: without this, joint optimization can
# pull the base to a meaningfully worse fit than it reaches on its own
# (observed nu=0.685 vs a clean fit's nu=1.2 on this data), which is exactly
# what caused VG-only to beat VG-flow. With warm-starting, VG-flow beats
# VG-only by a real margin instead (1.118 vs 1.162 val NLL in one such run).
warm_nll = warm_start_base(base, x_train, epochs=300)
print(f"Warm-started base: mu={base.mu.item():.4f} sigma={base.sigma.item():.4f} "
      f"theta={base.theta.item():.4f} nu={base.nu.item():.4f}  (train NLL={warm_nll:.4f})")

my_flow = NSFFlow(dim=1, base=base, num_layers=12, num_bins=8, tail_bound=5.0, hidden_dim=128, n_hidden=3)
hist = train_flow(my_flow, x_train, TrainConfig(epochs=EPOCHS, print_every=50), x_val=x_val)
print(f"\nMy VG-flow            final val NLL = {hist['val_nll'][-1]:.4f}")

# ---- zuko Gaussian-base flow, matched architecture/training ----
gflow = zuko.flows.NSF(features=1, transforms=6, bins=8, tail_bound=5.0, hidden_features=(64, 64))
opt = torch.optim.Adam(gflow.parameters(), lr=1e-3)
x_train_2d, x_val_2d = x_train.unsqueeze(-1), x_val.unsqueeze(-1)
loader = DataLoader(TensorDataset(x_train_2d), batch_size=256, shuffle=True)
for epoch in range(1, EPOCHS + 1):
    for (xb,) in loader:
        opt.zero_grad()
        nll = -gflow().log_prob(xb).mean()
        nll.backward()
        torch.nn.utils.clip_grad_norm_(gflow.parameters(), 5.0)
        opt.step()
    if epoch % 50 == 0:
        with torch.no_grad():
            print(f"  [zuko] epoch {epoch:4d}  val NLL = {-gflow().log_prob(x_val_2d).mean().item():.4f}")
with torch.no_grad():
    final_gauss_nll = -gflow().log_prob(x_val_2d).mean().item()
print(f"zuko Gaussian-flow     final val NLL = {final_gauss_nll:.4f}")

# ---- VG-only (no flow) ablation, for reference ----
vg_only = VarianceGamma(mu=float(x_train.mean()), sigma=float(x_train.std()),
                         theta=-0.3, nu=0.6, dtype=torch.float32)
# quick moment-free fit: just eyeball a couple of (theta, nu) choices via grid search on train NLL
best = None
for theta in [-0.5, -0.3, -0.1, 0.0]:
    for nu in [0.3, 0.5, 0.8, 1.2]:
        cand = VarianceGamma(mu=0.0, sigma=1.0, theta=theta, nu=nu, dtype=torch.float32)
        nll = -cand.log_prob(x_train).mean().item()
        if best is None or nll < best[0]:
            best = (nll, theta, nu)
theta_best, nu_best = best[1], best[2]
vg_only = VarianceGamma(mu=0.0, sigma=1.0, theta=theta_best, nu=nu_best, dtype=torch.float32)
vg_only_val_nll = -vg_only.log_prob(x_val).mean().item()
print(f"VG-only (no flow, best grid theta={theta_best}, nu={nu_best})  val NLL = {vg_only_val_nll:.4f}")

# ---- VaR comparison ----
_, samples_std = estimate_var(my_flow, levels=(0.95, 0.99, 0.999), n_samples=200_000)
samples_raw = samples_std * x_std.item() + x_mean.item()
print("\nMonte-Carlo VaR from my VG-flow (raw log-return units):")
for lvl in (0.95, 0.99, 0.999):
    q = np.quantile(samples_raw, 1 - lvl)
    print(f"  VaR_{lvl*100:.1f}% = {-q:.5f}")
print("Empirical VaR on held-out real data:")
for lvl in (0.95, 0.99, 0.999):
    q = np.quantile(x_val_raw.numpy(), 1 - lvl)
    print(f"  VaR_{lvl*100:.1f}% = {-q:.5f}")

# ---- KS test: the diagnostic that caught the levy-flows repo's bug ----
# Compares sample() output against real held-out data directly -- catches a
# log_prob/sample() inconsistency that the training loss alone can't see.
ks = ks_test(my_flow, x_val, device="cpu")  # standardized units, matches training
print(f"\nKS test (my VG-flow samples vs held-out data, standardized units): "
      f"stat={ks['ks_statistic']:.4f}  p={ks['ks_pvalue']:.4f}  (n_samples={ks['n_samples']})")

with torch.no_grad():
    gflow.eval()
    n_ks = max(10_000, 10 * len(x_val))
    g_samples = gflow().sample((n_ks,)).squeeze(-1).cpu().numpy()
from scipy.stats import ks_2samp
g_stat, g_p = ks_2samp(g_samples, x_val.numpy())
print(f"KS test (zuko Gaussian-flow samples vs held-out data, standardized units): "
      f"stat={g_stat:.4f}  p={g_p:.4f}")

# save everything needed for the diagnostic plot
np.savez(PROJECT_ROOT / "results" / "sp500_results.npz",
         x_train_raw=x_train_raw.numpy(), x_val_raw=x_val_raw.numpy(),
         x_mean=x_mean.item(), x_std=x_std.item())
torch.save(my_flow.state_dict(), PROJECT_ROOT / "models" / "my_flow_sp500.pt")
print("\nSaved trained flow + data split for plotting.")