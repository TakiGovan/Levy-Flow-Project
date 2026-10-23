"""
Variance Gamma (VG) base distribution for the flow.

Parameterization (Madan-Seneta / mixing representation), matching the
requirements exactly:

    X = mu + theta * G + sigma * sqrt(G) * Z,   G ~ Gamma(shape=1/nu, rate=1/nu), Z ~ N(0,1)

so E[G] = 1, Var[G] = nu. This is the standard VG process parameterization
used in Madan, Carr & Chang (1998).

The closed-form log-density involves the modified Bessel function of the
second kind K_v; PyTorch has no general-order K_v, so we wrap
scipy.special.kv in a custom autograd.Function. Only the gradient w.r.t. the
*argument* z is needed (the order v = 1/nu - 1/2 is a fixed hyperparameter of
the base distribution, not a learned tensor), and that gradient is exact via
the standard recurrence

    d/dz K_v(z) = -1/2 * (K_{v-1}(z) + K_{v+1}(z)).

This is what makes log_prob differentiable w.r.t. the flow's output z, which
is all that's required to backprop the NLL into the spline coupling layers.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import kv as scipy_kv


class BesselK(torch.autograd.Function):
    """
    Differentiable K_v(z), elementwise over z, for a (possibly-tensor, possibly
    learnable) order v.

    d/dz K_v(z) has an exact closed form via the standard recurrence
        K_v'(z) = -1/2 * (K_{v-1}(z) + K_{v+1}(z))
    d/dv K_v(z) has no simple closed form, so it's approximated with a central
    finite difference in v (v is always a scalar here -- the base distribution
    has one shared nu across the batch -- so this costs 2 extra scipy calls
    per forward, not per-element).
    """

    @staticmethod
    def forward(ctx, z: torch.Tensor, v: torch.Tensor, fd_eps: float = 1e-4):
        z_np = z.detach().cpu().numpy()
        v_val = float(v.detach().cpu())

        k_v = scipy_kv(v_val, z_np)
        k_vm1 = scipy_kv(v_val - 1.0, z_np)
        k_vp1 = scipy_kv(v_val + 1.0, z_np)
        k_v_plus_h = scipy_kv(v_val + fd_eps, z_np)
        k_v_minus_h = scipy_kv(v_val - fd_eps, z_np)
        dK_dv_np = (k_v_plus_h - k_v_minus_h) / (2 * fd_eps)

        ctx.save_for_backward(
            torch.as_tensor(k_vm1, dtype=z.dtype, device=z.device),
            torch.as_tensor(k_vp1, dtype=z.dtype, device=z.device),
            torch.as_tensor(dK_dv_np, dtype=z.dtype, device=z.device),
        )
        ctx.v_shape = v.shape
        ctx.v_requires_grad = v.requires_grad
        return torch.as_tensor(k_v, dtype=z.dtype, device=z.device)

    @staticmethod
    def backward(ctx, grad_output):
        k_vm1, k_vp1, dK_dv = ctx.saved_tensors
        dK_dz = -0.5 * (k_vm1 + k_vp1)
        grad_z = grad_output * dK_dz
        grad_v = None
        if ctx.v_requires_grad:
            grad_v = (grad_output * dK_dv).sum().reshape(ctx.v_shape)
        return grad_z, grad_v, None


def bessel_k(z: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return BesselK.apply(z, v)


def _inverse_softplus(x: float) -> float:
    """raw value r such that softplus(r) == x, for x > 0."""
    x = max(x, 1e-6)
    return math.log(math.expm1(x)) if x < 20 else x


class VarianceGamma(nn.Module):
    """
    Variance Gamma distribution, optionally with learnable parameters
    (mu, sigma, theta, nu) jointly trained alongside the flow via gradient
    descent on the NLL. sigma and nu are kept positive via a softplus
    reparameterization of an underlying unconstrained parameter; mu and theta
    are unconstrained.

    Set learnable=False to keep the original fixed-hyperparameter behavior.
    """

    def __init__(self, mu: float = 0.0, sigma: float = 1.0, theta: float = 0.0,
                 nu: float = 0.3, learnable: bool = True, dtype=torch.float32):
        super().__init__()
        assert sigma > 0 and nu > 0
        self.dtype = dtype
        self._min_sigma = 1e-3
        self._min_nu = 0.05  # keep away from nu->0, where the VG likelihood near x=mu is unbounded
                              # (a genuine scale-mixture degeneracy, not just a numerics issue --
                              # verified: unconstrained training drove nu down monotonically for many
                              # epochs and eventually hit a numerically fatal step in that regime)

        raw_sigma = _inverse_softplus(sigma - self._min_sigma)
        raw_nu = _inverse_softplus(nu - self._min_nu)

        mu_t = torch.tensor(float(mu), dtype=dtype)
        theta_t = torch.tensor(float(theta), dtype=dtype)
        raw_sigma_t = torch.tensor(float(raw_sigma), dtype=dtype)
        raw_nu_t = torch.tensor(float(raw_nu), dtype=dtype)

        if learnable:
            self.mu_ = nn.Parameter(mu_t)
            self.theta_ = nn.Parameter(theta_t)
            self.raw_sigma_ = nn.Parameter(raw_sigma_t)
            self.raw_nu_ = nn.Parameter(raw_nu_t)
        else:
            self.register_buffer("mu_", mu_t)
            self.register_buffer("theta_", theta_t)
            self.register_buffer("raw_sigma_", raw_sigma_t)
            self.register_buffer("raw_nu_", raw_nu_t)

    # -- constrained parameter views -------------------------------------
    @property
    def mu(self) -> torch.Tensor:
        return self.mu_

    @property
    def theta(self) -> torch.Tensor:
        return self.theta_

    @property
    def sigma(self) -> torch.Tensor:
        return F.softplus(self.raw_sigma_) + self._min_sigma

    @property
    def nu(self) -> torch.Tensor:
        return F.softplus(self.raw_nu_) + self._min_nu

    def log_prob(self, x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        mu, sigma, theta, nu = self.mu, self.sigma, self.theta, self.nu
        xc = x - mu
        A = 2 * sigma**2 / nu + theta**2  # scalar tensor

        z = torch.abs(xc) * torch.sqrt(A) / sigma**2
        z = torch.clamp(z, min=eps)  # K_v(0) diverges; xc==0 has probability zero
        v = 1.0 / nu - 0.5  # Bessel order; tracks nu's gradient via finite-diff in BesselK

        log_const = (
            math.log(2.0)
            - torch.log(sigma)
            - 0.5 * math.log(2 * math.pi)
            - (1.0 / nu) * torch.log(nu)
            - torch.lgamma(1.0 / nu)
        )
        log_pow = (1.0 / (2 * nu) - 0.25) * torch.log(torch.clamp(xc**2, min=eps) / A)
        logK = torch.log(bessel_k(z, v))

        return log_const + theta * xc / sigma**2 + log_pow + logK

    @torch.no_grad()
    def rsample(self, shape) -> torch.Tensor:
        mu, sigma, theta, nu = self.mu.item(), self.sigma.item(), self.theta.item(), self.nu.item()
        device = self.mu_.device
        shape = tuple(shape) if not isinstance(shape, int) else (shape,)
        g = np.random.gamma(shape=1.0 / nu, scale=nu, size=shape)  # E[g] = 1
        z = np.random.standard_normal(size=shape)
        x = mu + theta * g + sigma * np.sqrt(g) * z
        return torch.as_tensor(x, dtype=self.dtype, device=device)