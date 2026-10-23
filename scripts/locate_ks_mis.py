"""
Locate exactly where a KS-test mismatch between model samples and real data
occurs -- bulk/body vs tail -- rather than just reporting the statistic.

Run this after real_data_compare.py has saved my_flow_sp500.pt + sp500_results.npz.
"""

from pathlib import Path

import numpy as np
import torch

from levy_flow import NSFFlow, VarianceGamma

PROJECT_ROOT = Path(__file__).resolve().parents[1]

d = np.load(PROJECT_ROOT / "results" / "sp500_results.npz")
x_val_raw = d["x_val_raw"]
x_mean, x_std = float(d["x_mean"]), float(d["x_std"])
x_val = (x_val_raw - x_mean) / x_std

base = VarianceGamma(mu=0.0, sigma=1.0, theta=0.0, nu=0.5, dtype=torch.float32)
flow = NSFFlow(dim=1, base=base, num_layers=12, num_bins=8, tail_bound=5.0, hidden_dim=128, n_hidden=3)
flow.load_state_dict(torch.load(PROJECT_ROOT / "models" / "my_flow_sp500.pt"))
flow.eval()

with torch.no_grad():
    n_samples = max(50_000, 10 * len(x_val))
    samples = flow.sample(n_samples).numpy().flatten()

grid = np.linspace(min(x_val.min(), samples.min()), max(x_val.max(), samples.max()), 4000)
cdf_data = np.searchsorted(np.sort(x_val), grid, side="right") / len(x_val)
cdf_model = np.searchsorted(np.sort(samples), grid, side="right") / len(samples)
diff = np.abs(cdf_data - cdf_model)
i = np.argmax(diff)

print(f"n_val = {len(x_val)}, n_samples = {n_samples}")
print(f"\nmax |CDF_data - CDF_model| = {diff[i]:.4f} at standardized x = {grid[i]:.3f}"
      f"  (raw return = {grid[i]*x_std + x_mean:.4f})")
print(f"CDF_data at that point:  {cdf_data[i]:.4f}")
print(f"CDF_model at that point: {cdf_model[i]:.4f}")

tail_bound_proxy = 4.0  # a bit inside the spline's tail_bound=5.0, where identity tails start dominating
tail_mask = np.abs(grid) > tail_bound_proxy
body_mask = ~tail_mask
print(f"\nmax deviation restricted to |x| > {tail_bound_proxy} (near/beyond tail_bound): "
      f"{diff[tail_mask].max() if tail_mask.any() else float('nan'):.4f}")
print(f"max deviation restricted to |x| <= {tail_bound_proxy} (bulk/body): {diff[body_mask].max():.4f}")

# also report deviation specifically around the deepest tails, where crisis days live
deep_tail_mask = grid < -4.0
if deep_tail_mask.any():
    print(f"max deviation in x < -4.0 (crash-day region): {diff[deep_tail_mask].max():.4f}")