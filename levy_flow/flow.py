"""
Neural Spline Flow with a Variance-Gamma base distribution.

Design notes
------------
- dim == 1 (e.g. a single return series): there's no "other half" of the
  vector to condition on, so each layer's spline parameters are free,
  directly-learned nn.Parameters (an unconditional monotonic transform).
  Stacking several such layers gives an expressive univariate flow -- this
  is the standard trick for 1-D normalizing flows.
- dim > 1: standard NSF coupling layers. The active/passive split alternates
  every layer (like RealNVP/Glow-style checkerboard masking on a vector),
  and an MLP conditioner maps the passive half to spline parameters for the
  active half.

Either way, log p_X(x) = log p_Z(T(x)) + log|det dT/dx|, with T the composed
flow and p_Z the VarianceGamma base. Training minimizes -mean(log p_X(x)).
"""

from typing import List
import torch
import torch.nn as nn

from .splines import unconstrained_rational_quadratic_spline
from .variance_gamma import VarianceGamma

PARAMS_PER_BIN = 3  # unnormalized width, height, (num_bins - 1 shared later) derivative


def _params_per_dim(num_bins: int) -> int:
    # widths: num_bins, heights: num_bins, internal derivatives: num_bins - 1
    return 3 * num_bins - 1


def _split_params(raw: torch.Tensor, num_bins: int):
    w, h, d = torch.split(raw, [num_bins, num_bins, num_bins - 1], dim=-1)
    return w, h, d


class MLPConditioner(nn.Module):
    """Passive dims -> spline parameters for active dims."""

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 64, n_hidden: int = 2):
        super().__init__()
        layers: List[nn.Module] = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_hidden - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers += [nn.Linear(hidden_dim, out_dim)]
        self.net = nn.Sequential(*layers)
        # zero-init last layer -> flow starts close to identity, stabilizes early training
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x_passive: torch.Tensor) -> torch.Tensor:
        return self.net(x_passive)


class NSFLayer(nn.Module):
    """One coupling (or, for dim==1, unconditional) rational-quadratic spline layer."""

    def __init__(self, dim: int, mask: torch.Tensor, num_bins: int = 8,
                 tail_bound: float = 5.0, hidden_dim: int = 64, n_hidden: int = 2):
        super().__init__()
        self.dim = dim
        self.num_bins = num_bins
        self.tail_bound = tail_bound
        self.register_buffer("active_mask", mask.bool())  # True = transformed this layer

        n_active = int(mask.sum().item())
        n_passive = dim - n_active
        out_dim = n_active * _params_per_dim(num_bins)

        if n_passive == 0:
            # dim == 1 (or a layer with everything active): unconditional, free parameters
            self.conditioner = None
            self.unconditional_params = nn.Parameter(0.01 * torch.randn(out_dim))
        else:
            self.conditioner = MLPConditioner(n_passive, out_dim, hidden_dim, n_hidden)
            self.unconditional_params = None

    def _get_raw_params(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        n_active = int(self.active_mask.sum().item())
        if self.conditioner is None:
            raw = self.unconditional_params.expand(batch, -1)
        else:
            x_passive = x[:, ~self.active_mask]
            raw = self.conditioner(x_passive)
        return raw.reshape(batch, n_active, _params_per_dim(self.num_bins))

    def forward(self, x: torch.Tensor, inverse: bool = False):
        """x: (batch, dim) -> (y, logabsdet) with logabsdet: (batch,)"""
        raw = self._get_raw_params(x)
        uw, uh, ud = _split_params(raw, self.num_bins)

        x_active = x[:, self.active_mask]
        y_active, logdet_active = unconstrained_rational_quadratic_spline(
            inputs=x_active, unnormalized_widths=uw, unnormalized_heights=uh,
            unnormalized_derivatives=ud, inverse=inverse, tail_bound=self.tail_bound,
        )

        y = x.clone()
        y[:, self.active_mask] = y_active
        logabsdet = logdet_active.sum(dim=-1)
        return y, logabsdet


class NSFFlow(nn.Module):
    """Stack of NSFLayer transforms, plus a VarianceGamma base distribution."""

    def __init__(self, dim: int, base: VarianceGamma, num_layers: int = 6,
                 num_bins: int = 8, tail_bound: float = 5.0,
                 hidden_dim: int = 64, n_hidden: int = 2):
        super().__init__()
        self.dim = dim
        self.base = base
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if dim == 1:
                mask = torch.tensor([True])  # every layer transforms the single dim
            else:
                mask = torch.zeros(dim, dtype=torch.bool)
                mask[i % 2::2] = True  # alternating checkerboard-style split
            self.layers.append(NSFLayer(dim, mask, num_bins, tail_bound, hidden_dim, n_hidden))

    def forward(self, x: torch.Tensor):
        """Data -> base space. Returns (z, total_logabsdet)."""
        logabsdet = x.new_zeros(x.shape[0])
        z = x
        for layer in self.layers:
            z, ld = layer(z, inverse=False)
            logabsdet = logabsdet + ld
        return z, logabsdet

    def inverse(self, z: torch.Tensor):
        """Base space -> data. Returns (x, total_logabsdet) [logabsdet of the inverse map]."""
        logabsdet = z.new_zeros(z.shape[0])
        x = z
        for layer in reversed(self.layers):
            x, ld = layer(x, inverse=True)
            logabsdet = logabsdet + ld
        return x, logabsdet

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        z, logabsdet = self.forward(x)
        base_lp = self.base.log_prob(z.squeeze(-1) if self.dim == 1 else z)
        if self.dim == 1:
            base_lp = base_lp.reshape(-1)
        else:
            base_lp = base_lp.sum(dim=-1)
        return base_lp + logabsdet

    @torch.no_grad()
    def sample(self, n: int) -> torch.Tensor:
        z = self.base.rsample((n,) if self.dim == 1 else (n, self.dim))
        if self.dim == 1:
            z = z.reshape(n, 1)
        x, _ = self.inverse(z)
        return x