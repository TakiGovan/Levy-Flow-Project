# Levy Flow

This project is an experiment with normalizing flows for return data. The main
model is a one-dimensional neural spline flow with a Variance-Gamma base. The
goal so far has been to see whether the flow improves the fit where it matters
for risk work: the tails, not just the average likelihood.

## Project layout

- `levy_flow/` — the spline flow, Variance-Gamma distribution, training helpers,
  and the **work-in-progress** copula code.
- `scripts/` — runnable examples and S&P 500 analysis scripts.
- `notebooks/` — training notebooks and the saved VG-only vs. VG-flow analysis.
- `data/` — input return data.
- `models/` — saved model weights.
- `results/` — saved plots and the train/validation split used by the analysis.
- `src/` and `flows/` — the earlier flow implementation and its notebooks; kept
  separate from the newer `levy_flow` code.

## Current results

The saved comparison in
[`notebooks/vg_flow_vs_vg_only_analysis.ipynb`](notebooks/vg_flow_vs_vg_only_analysis.ipynb)
uses a chronological train/validation split. In that run, the VG-only model and
VG-flow had very similar overall validation NLLs (1.3983 and 1.3954). Looking
only at that number hides the tail difference: the tail NLL was 8.5943 for
VG-only and 8.0033 for VG-flow, based on 13 validation observations in the
tail region.

| Validation measure | VG-only | VG-flow |
| --- | ---: | ---: |
| Overall NLL | 1.3983 | 1.3954 |
| Tail NLL (13 observations) | 8.5943 | 8.0033 |
| KS statistic | 0.0636 | 0.0545 |
| KS p-value | 0.0341 | 0.1115 |

VaR was mixed, not a clean win at every level. On standardized returns, the
empirical, VG-only, and VG-flow VaR values were 1.3825, 1.6116, and 1.5283 at
95%; 2.3924, 2.8645, and 3.0447 at 99%; and 4.7957, 4.7111, and 4.9534 at
99.9%. These are useful signals, but it's one split and the tail sample is
small, so the result isn't settled.

## Run the analysis

Run commands from the project root. The real-data comparison trains models and
saves its split and weights under `results/` and `models/`:

```powershell
python -m scripts.real_data_compare
```

Then make the validation plots or inspect where the KS mismatch occurs:

```powershell
python -m scripts.plot_sp500_diagnostics
python -m scripts.locate_ks_mis
```

For a synthetic end-to-end example:

```powershell
python -m scripts.demo
```

The analysis notebook expects the saved artifacts from
`scripts.real_data_compare`.

## Model notes

The implemented, tested use case is one-dimensional returns. The spline uses
identity tails outside its fitted interval, and the Variance-Gamma base uses
the normal-gamma mixture representation. Training standardizes returns first;
VaR samples are converted back to raw return units.

The multivariate/copula work is deliberately unfinished. There is code for
Gaussian and Student-t copulas, but it has not been validated as a complete
multivariate risk model. Treat it as a place to continue the work, not as a
supported feature.
