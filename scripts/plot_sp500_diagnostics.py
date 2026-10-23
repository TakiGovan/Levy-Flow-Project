import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

import torch

from levy_flow import NSFFlow, VarianceGamma

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# reload the trained flow + the exact standardized train split it was fit on
d = np.load(PROJECT_ROOT / "results" / "sp500_results.npz")
x_train_raw = d["x_train_raw"]
x_mean, x_std = float(d["x_mean"]), float(d["x_std"])
train_data = (x_train_raw - x_mean) / x_std  # standardized, same units the flow was trained on

base = VarianceGamma(mu=0.0, sigma=1.0, theta=0.0, nu=0.5, dtype=torch.float32)
model = NSFFlow(dim=1, base=base, num_layers=12, num_bins=8, tail_bound=5.0, hidden_dim=128, n_hidden=3)
model.load_state_dict(torch.load(PROJECT_ROOT / "models" / "my_flow_sp500.pt"))
model.eval()

with torch.no_grad():
    samples = model.sample(10000).numpy().flatten()

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0, 0].hist(train_data, bins=100, density=True, alpha=0.5, label='S&P 500 data (standardized)')
axes[0, 0].hist(samples, bins=100, density=True, alpha=0.5, label='Lévy-Flow (VG)')
axes[0, 0].set_title('Density Comparison')
axes[0, 0].legend()

# Log scale tails
axes[0, 1].hist(train_data, bins=100, density=True, alpha=0.5, log=True, label='S&P 500 data')
axes[0, 1].hist(samples, bins=100, density=True, alpha=0.5, log=True, label='Lévy-Flow (VG)')
axes[0, 1].set_title('Tails (Log Scale)')
axes[0, 1].legend()

# QQ plot
sorted_data = np.sort(train_data)
sorted_samples = np.sort(samples[:len(train_data)])
axes[1, 0].scatter(sorted_data, sorted_samples, s=1, alpha=0.5)
lo, hi = sorted_data.min(), sorted_data.max()
axes[1, 0].plot([lo, hi], [lo, hi], 'r--')
axes[1, 0].set_title('QQ Plot')
axes[1, 0].set_xlabel('Data Quantiles')
axes[1, 0].set_ylabel('Model Quantiles')

# CDF
axes[1, 1].hist(train_data, bins=100, density=True, cumulative=True, alpha=0.5, label='S&P 500 data')
axes[1, 1].hist(samples, bins=100, density=True, cumulative=True, alpha=0.5, label='Lévy-Flow (VG)')
axes[1, 1].set_title('CDF Comparison')
axes[1, 1].legend()

plt.tight_layout()
output_path = PROJECT_ROOT / "results" / "sp500_vg_flow_validation.png"
plt.savefig(output_path, dpi=150)
print(f"saved {output_path}")
