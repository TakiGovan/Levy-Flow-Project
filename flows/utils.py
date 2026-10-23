"""
Variance Gamma Distribution for Normalizing Flows.

The Variance Gamma (VG) distribution is a pure jump Lévy process widely used
in financial modeling. It can be represented as a time-changed Brownian motion
where time follows a Gamma process.

Key properties:
- Semi-heavy tails (heavier than Gaussian, lighter than Cauchy)
- Can capture skewness and excess kurtosis
- Closed-form PDF available via modified Bessel function
- Natural for modeling asset returns with jumps

References:
- Madan, D.B., Carr, P., & Chang, E.C. (1998). The Variance Gamma Process
  and Option Pricing. Review of Finance.
- Madan, D.B., & Seneta, E. (1990). The Variance Gamma Model for Share
  Market Returns. Journal of Business.
"""

import math
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.special import gammaln


class LevyDistribution(nn.Module, ABC):
    """
    Abstract base class for Lévy distributions used as flow bases.

    All Lévy distributions must implement:
    - log_prob: Compute log probability density
    - sample: Generate samples via reparameterization trick
    - characteristic_function: Compute the characteristic function (for CGMY)

    Properties:
    - tail_index: The tail index α determining power-law decay P(X > x) ~ x^(-α)
    - is_symmetric: Whether the distribution is symmetric around its location
    """

    def __init__(self, dim: int = 1, learnable: bool = True):
        """
        Args:
            dim: Dimensionality of the distribution
            learnable: Whether parameters should be learnable
        """
        super().__init__()
        self.dim = dim
        self.learnable = learnable

    @abstractmethod
    def log_prob(self, x: Tensor) -> Tensor:
        """
        Compute log probability density.

        Args:
            x: Input tensor of shape (batch_size, dim)

        Returns:
            Log probabilities of shape (batch_size,)
        """
        pass

    @abstractmethod
    def sample(self, num_samples: int) -> Tensor:
        """
        Generate samples using reparameterization trick for gradient flow.

        Args:
            num_samples: Number of samples to generate

        Returns:
            Samples of shape (num_samples, dim)
        """
        pass

    @abstractmethod
    def rsample(self, num_samples: int) -> Tensor:
        """
        Generate reparameterized samples (allows gradients through sampling).

        Args:
            num_samples: Number of samples to generate

        Returns:
            Samples of shape (num_samples, dim)
        """
        pass

    @property
    @abstractmethod
    def tail_index(self) -> float:
        """
        Return the tail index α of the distribution.

        For power-law tails: P(X > x) ~ x^(-α) as x → ∞

        - Gaussian: α = ∞ (exponential decay, no power-law tail)
        - Student-t(ν): α = ν
        - Variance Gamma: α depends on parameters
        - NIG: α = 1 (semi-heavy tails)
        - CGMY: α = Y (the Y parameter)
        - Stable(α): α = α (the stability parameter)
        """
        pass

    @property
    def is_symmetric(self) -> bool:
        """Whether the distribution is symmetric around location."""
        return True  # Override in subclasses if asymmetric

    def characteristic_function(self, t: Tensor) -> Tensor:
        """
        Compute the characteristic function φ(t) = E[exp(itX)].

        Useful for CGMY and other distributions without closed-form PDFs.

        Args:
            t: Frequency tensor

        Returns:
            Complex tensor of characteristic function values
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement characteristic_function"
        )

    def cdf(self, x: Tensor) -> Tensor:
        """
        Compute cumulative distribution function.

        Args:
            x: Input tensor

        Returns:
            CDF values
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement cdf"
        )

    def icdf(self, p: Tensor) -> Tensor:
        """
        Compute inverse CDF (quantile function).

        Args:
            p: Probability values in [0, 1]

        Returns:
            Quantile values
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement icdf"
        )

    def entropy(self) -> Tensor:
        """Compute differential entropy of the distribution."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement entropy"
        )

    def kl_divergence(self, other: "LevyDistribution") -> Tensor:
        """
        Compute KL divergence KL(self || other).

        Args:
            other: Another Lévy distribution

        Returns:
            KL divergence value
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement kl_divergence"
        )

    def get_params(self) -> dict:
        """Return dictionary of distribution parameters."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement get_params"
        )

    def __repr__(self) -> str:
        params = self.get_params()
        param_str = ", ".join(f"{k}={v:.4f}" for k, v in params.items())
        return f"{self.__class__.__name__}({param_str})"

    

def _log_bessel_k(v: Tensor, z: Tensor) -> Tensor:
    """
    Compute log of modified Bessel function K_v(z) using pure PyTorch.

    Uses asymptotic expansion for large z and series expansion for small z.
    Fully differentiable and GPU-compatible.

    K_v(z) ~ sqrt(pi/(2z)) * exp(-z) * (1 + (4v^2-1)/(8z) + ...)  for large z

    Args:
        v: Order of Bessel function, shape (dim,) or scalar
        z: Argument, shape (batch, dim) or (batch,)

    Returns:
        log K_v(z), same shape as z
    """
    # Ensure z is positive and bounded away from 0
    z = torch.clamp(z, min=1e-8)

    # For numerical stability, we use different approximations based on z
    # Large z (> 10): asymptotic expansion
    # Small z: use a rational approximation or series

    # Asymptotic expansion for large z:
    # log K_v(z) ≈ log(sqrt(pi/(2z))) - z + log(1 + (4v^2-1)/(8z) + ...)
    #            = 0.5*log(pi/2) - 0.5*log(z) - z + correction terms

    # Broadcast v to match z shape if needed
    if v.dim() == 1 and z.dim() == 2:
        v = v.unsqueeze(0)  # (1, dim)

    v2 = v ** 2

    # Leading asymptotic term (valid for all z > 0, more accurate for large z)
    log_leading = 0.5 * math.log(math.pi / 2) - 0.5 * torch.log(z) - z

    # Correction terms for better accuracy
    # (4v^2 - 1) / (8z) + (4v^2-1)(4v^2-9) / (2!(8z)^2) + ...
    mu = 4 * v2
    term1 = (mu - 1) / (8 * z)
    term2 = (mu - 1) * (mu - 9) / (128 * z ** 2)
    term3 = (mu - 1) * (mu - 9) * (mu - 25) / (3072 * z ** 3)

    # Log of correction factor: log(1 + term1 + term2 + term3 + ...)
    # For numerical stability, compute this carefully
    correction_sum = 1 + term1 + term2 + term3
    correction_sum = torch.clamp(correction_sum, min=1e-10)  # Ensure positive
    log_correction = torch.log(correction_sum)

    # Full asymptotic result
    log_k_asymptotic = log_leading + log_correction

    # For small z, we need a different approach
    # For |v| < 0.5 and small z, K_v(z) ≈ Gamma(v) * (z/2)^(-v) / 2  for v > 0
    # and K_0(z) ≈ -log(z/2) - gamma for v = 0

    # Use a smooth blend between small-z and large-z approximations
    # The asymptotic expansion is accurate for z > 2*|v| roughly

    # For small z with v > 0:
    # K_v(z) ≈ 0.5 * Gamma(v) * (z/2)^(-v)
    # log K_v(z) ≈ log(0.5) + gammaln(v) - v * log(z/2)
    #            = log(0.5) + gammaln(v) - v * (log(z) - log(2))
    #            = -log(2) + gammaln(v) - v * log(z) + v * log(2)
    #            = gammaln(v) + (v - 1) * log(2) - v * log(z)

    # For v near 0, K_0(z) ≈ -log(z/2) - gamma (Euler's constant)
    # We'll use a blend

    v_abs = torch.abs(v)
    v_safe = torch.clamp(v_abs, min=0.1)  # Avoid v=0 singularity in gammaln

    # Small z approximation (valid for z << 1 and v > 0)
    log_k_small = gammaln(v_safe) + (v_safe - 1) * math.log(2) - v_safe * torch.log(z)

    # Blend based on z value - use asymptotic for z > threshold
    # Threshold depends on v, but z > 10 is generally safe
    threshold = 2.0 + 2 * v_abs
    blend_weight = torch.sigmoid((z - threshold) * 2)  # Smooth transition

    log_k = blend_weight * log_k_asymptotic + (1 - blend_weight) * log_k_small

    # Clamp final result to avoid extreme values
    log_k = torch.clamp(log_k, min=-100, max=100)

    return log_k


class VarianceGamma(LevyDistribution):
    """
    Variance Gamma distribution parameterized by (μ, σ, θ, ν).

    The VG distribution has density:
        p(x) = (2 / (σ√(2π) ν^(1/ν) Γ(1/ν))) *
               ((|x - μ|) / √(2σ²/ν + θ²))^(1/ν - 1/2) *
               K_{1/ν - 1/2}(|x - μ| √(2σ²/ν + θ²) / σ²) *
               exp(θ(x - μ) / σ²)

    where K_ν is the modified Bessel function of the second kind.

    Parameters:
        μ (mu): Location parameter
        σ (sigma): Volatility parameter (> 0)
        θ (theta): Skewness parameter (θ < 0 for left skew, θ > 0 for right skew)
        ν (nu): Kurtosis/variance of time change (> 0, smaller = heavier tails)

    Special cases:
        - θ = 0: Symmetric VG
        - ν → 0: Approaches Gaussian
        - Large ν: Heavier tails, more jump-like behavior
    """

    def __init__(
        self,
        dim: int = 1,
        mu: float = 0.0,
        sigma: float = 1.0,
        theta: float = 0.0,
        nu: float = 1.0,
        learnable: bool = True,
    ):
        """
        Initialize Variance Gamma distribution.

        Args:
            dim: Dimensionality (independent VG marginals)
            mu: Location parameter
            sigma: Volatility parameter (> 0)
            theta: Skewness parameter
            nu: Kurtosis parameter (> 0)
            learnable: Whether parameters are learnable
        """
        super().__init__(dim=dim, learnable=learnable)

        # Store initial values for reset
        self._init_mu = mu
        self._init_sigma = sigma
        self._init_theta = theta
        self._init_nu = nu

        if learnable:
            # Unconstrained parameters for optimization
            self._mu = nn.Parameter(torch.tensor([mu] * dim, dtype=torch.float32))
            self._log_sigma = nn.Parameter(
                torch.tensor([math.log(sigma)] * dim, dtype=torch.float32)
            )
            self._theta = nn.Parameter(torch.tensor([theta] * dim, dtype=torch.float32))
            self._log_nu = nn.Parameter(
                torch.tensor([math.log(nu)] * dim, dtype=torch.float32)
            )
        else:
            self.register_buffer("_mu", torch.tensor([mu] * dim, dtype=torch.float32))
            self.register_buffer(
                "_log_sigma", torch.tensor([math.log(sigma)] * dim, dtype=torch.float32)
            )
            self.register_buffer("_theta", torch.tensor([theta] * dim, dtype=torch.float32))
            self.register_buffer(
                "_log_nu", torch.tensor([math.log(nu)] * dim, dtype=torch.float32)
            )

    @property
    def mu(self) -> Tensor:
        """Location parameter."""
        return self._mu

    @property
    def sigma(self) -> Tensor:
        """Volatility parameter (positive)."""
        return torch.exp(self._log_sigma)

    @property
    def theta(self) -> Tensor:
        """Skewness parameter."""
        return self._theta

    @property
    def nu(self) -> Tensor:
        """Kurtosis parameter (positive)."""
        return torch.exp(self._log_nu)

    @property
    def tail_index(self) -> float:
        """
        VG has semi-heavy tails with effective tail index.

        The VG distribution has exponentially decaying tails modified by
        power-law prefactor. The effective tail behavior is controlled by ν.
        Larger ν = slower exponential decay = effectively heavier tails.

        Returns approximately 2/ν for practical purposes in the intermediate regime.
        """
        nu_val = self.nu.mean().item()
        # VG tails decay as |x|^(1/ν - 1) * exp(-c|x|)
        # For practical finite-sample behavior, return 2/ν
        return 2.0 / nu_val

    @property
    def is_symmetric(self) -> bool:
        """Check if distribution is symmetric (theta ≈ 0)."""
        return torch.allclose(self.theta, torch.zeros_like(self.theta), atol=1e-6)

    def _compute_vg_params(self) -> tuple:
        """Compute derived parameters for VG density."""
        sigma = self.sigma
        theta = self.theta
        nu = self.nu

        # Derived parameters
        sigma2 = sigma ** 2
        theta2 = theta ** 2

        # α = 1/ν - 1/2 (order of Bessel function)
        alpha = 1.0 / nu - 0.5

        # β = √(2σ²/ν + θ²) / σ²
        beta = torch.sqrt(2 * sigma2 / nu + theta2) / sigma2

        # γ = θ / σ²
        gamma = theta / sigma2

        return alpha, beta, gamma, sigma2, nu

    def log_prob(self, x: Tensor) -> Tensor:
        """
        Compute log probability density of VG distribution.

        Uses the representation:
        log p(x) = log(2) - log(σ√(2π)) - log(Γ(1/ν)) - (1/ν)log(ν)
                   + (1/ν - 1/2) log(|x - μ| / β)
                   + log(K_{1/ν - 1/2}(β|x - μ|))
                   + γ(x - μ)

        Fully differentiable and GPU-compatible using pure PyTorch Bessel K
        approximation.

        Args:
            x: Input tensor of shape (batch_size, dim) or (batch_size,)

        Returns:
            Log probabilities of shape (batch_size,)
        """
        if x.dim() == 1:
            x = x.unsqueeze(-1)

        # Get parameters - keep on same device as x
        mu = self.mu.to(x.device)
        sigma = self.sigma.to(x.device)
        theta = self.theta.to(x.device)
        nu = self.nu.to(x.device)

        sigma2 = sigma ** 2
        inv_nu = 1.0 / nu

        # Bessel function order
        bessel_order = inv_nu - 0.5

        # Scaling parameter
        c = torch.sqrt(2.0 / nu + (theta ** 2) / sigma2)

        # Centered and scaled input
        x_centered = x - mu
        x_abs = torch.abs(x_centered) + 1e-8  # Numerical stability

        # Log normalization constant
        log_norm = (
            math.log(2)
            - 0.5 * math.log(2 * math.pi)
            - torch.log(sigma)
            - gammaln(inv_nu)
            - inv_nu * torch.log(nu)
        )

        # Bessel function argument
        bessel_arg = c * x_abs

        # Log of |x - μ|^(1/ν - 1/2) term
        log_power = bessel_order * torch.log(x_abs / c)

        # Exponential tilt term
        exp_term = theta * x_centered / sigma2

        # Log Bessel function using pure PyTorch (differentiable, GPU-compatible)
        log_bessel = _log_bessel_k(bessel_order, bessel_arg)

        # Combine all terms
        log_p = log_norm + log_power + log_bessel + exp_term

        # Clamp to avoid -inf (very unlikely points)
        log_p = torch.clamp(log_p, min=-100)

        # Sum over dimensions (independent marginals)
        return log_p.sum(dim=-1)

    def sample(self, num_samples: int) -> Tensor:
        """
        Sample from VG distribution using time-changed Brownian motion.

        VG can be sampled as: X = μ + θG + σ√G * Z
        where G ~ Gamma(1/ν, 1/ν) and Z ~ N(0, 1)

        Args:
            num_samples: Number of samples

        Returns:
            Samples of shape (num_samples, dim)
        """
        return self.rsample(num_samples)

    def rsample(self, num_samples: int) -> Tensor:
        """
        Reparameterized sampling for gradient flow.

        Uses the representation: X = μ + θG + σ√G * Z
        where G ~ Gamma(shape=1/ν, rate=1/ν) and Z ~ N(0, 1)

        The Gamma is reparameterized using implicit gradients.
        """
        device = self.mu.device
        dtype = self.mu.dtype

        nu = self.nu
        mu = self.mu
        theta = self.theta
        sigma = self.sigma

        # Shape and rate for Gamma distribution
        # G ~ Gamma(1/ν, 1/ν) has mean 1 and variance ν
        shape = 1.0 / nu
        rate = 1.0 / nu

        # Sample Gamma using PyTorch's reparameterized sampler
        gamma_dist = torch.distributions.Gamma(shape, rate)
        G = gamma_dist.rsample((num_samples,))  # (num_samples, dim)

        # Sample standard normal
        Z = torch.randn(num_samples, self.dim, device=device, dtype=dtype)

        # VG sample: X = μ + θG + σ√G * Z
        samples = mu + theta * G + sigma * torch.sqrt(G) * Z

        return samples

    def characteristic_function(self, t: Tensor) -> Tensor:
        """
        Compute characteristic function of VG distribution.

        φ(t) = exp(iμt) * (1 - iθνt + σ²νt²/2)^(-1/ν)

        Args:
            t: Frequency tensor

        Returns:
            Complex tensor of CF values
        """
        mu = self.mu
        sigma = self.sigma
        theta = self.theta
        nu = self.nu

        # Complex exponential for location shift
        phase = torch.complex(torch.zeros_like(t), mu * t)
        exp_phase = torch.exp(phase)

        # Lévy-Khintchine exponent for VG
        # (1 - iθνt + σ²νt²/2)^(-1/ν)
        real_part = 1.0 + 0.5 * (sigma ** 2) * nu * (t ** 2)
        imag_part = -theta * nu * t

        base = torch.complex(real_part, imag_part)
        exponent = -1.0 / nu

        cf = exp_phase * torch.pow(base, exponent)

        return cf

    def get_params(self) -> dict:
        """Return dictionary of distribution parameters."""
        return {
            "mu": self.mu.mean().item(),
            "sigma": self.sigma.mean().item(),
            "theta": self.theta.mean().item(),
            "nu": self.nu.mean().item(),
        }

    def reset_parameters(self):
        """Reset parameters to initial values."""
        with torch.no_grad():
            self._mu.fill_(self._init_mu)
            self._log_sigma.fill_(math.log(self._init_sigma))
            self._theta.fill_(self._init_theta)
            self._log_nu.fill_(math.log(self._init_nu))



class StudentT(LevyDistribution):
    """
    Student-t distribution parameterized by (nu, mu, sigma).

    The density is:
        p(x) = Gamma((nu+1)/2) / (Gamma(nu/2) * sqrt(nu*pi) * sigma)
               * (1 + ((x-mu)/sigma)^2 / nu)^(-(nu+1)/2)

    Parameters:
        nu: Degrees of freedom (nu > 2 for finite variance)
        mu: Location parameter
        sigma: Scale parameter (sigma > 0)
    """

    def __init__(
        self,
        dim: int = 1,
        nu: float = 3.0,
        mu: float = 0.0,
        sigma: float = 1.0,
        learnable: bool = True,
    ):
        super().__init__(dim=dim, learnable=learnable)

        assert nu > 0, "nu must be positive"
        assert sigma > 0, "sigma must be positive"

        self._init_nu = nu
        self._init_mu = mu
        self._init_sigma = sigma

        if learnable:
            # Store log(nu) for unconstrained optimization, enforce nu > 2
            self._log_nu = nn.Parameter(
                torch.tensor([math.log(nu)] * dim, dtype=torch.float32)
            )
            self._mu = nn.Parameter(torch.tensor([mu] * dim, dtype=torch.float32))
            self._log_sigma = nn.Parameter(
                torch.tensor([math.log(sigma)] * dim, dtype=torch.float32)
            )
        else:
            self.register_buffer(
                "_log_nu", torch.tensor([math.log(nu)] * dim, dtype=torch.float32)
            )
            self.register_buffer("_mu", torch.tensor([mu] * dim, dtype=torch.float32))
            self.register_buffer(
                "_log_sigma", torch.tensor([math.log(sigma)] * dim, dtype=torch.float32)
            )

    @property
    def nu(self) -> Tensor:
        """Degrees of freedom (positive)."""
        return torch.exp(self._log_nu)

    @property
    def mu(self) -> Tensor:
        """Location parameter."""
        return self._mu

    @property
    def sigma(self) -> Tensor:
        """Scale parameter (positive)."""
        return torch.exp(self._log_sigma)

    @property
    def tail_index(self) -> float:
        """Student-t has power-law tail index equal to nu."""
        return self.nu.mean().item()

    @property
    def is_symmetric(self) -> bool:
        return True

    def log_prob(self, x: Tensor) -> Tensor:
        """
        Compute log probability density.

        log p(x) = log Gamma((nu+1)/2) - log Gamma(nu/2)
                   - 0.5*log(nu*pi) - log(sigma)
                   - (nu+1)/2 * log(1 + ((x-mu)/sigma)^2 / nu)
        """
        if x.dim() == 1:
            x = x.unsqueeze(-1)

        nu = self.nu.to(x.device)
        mu = self.mu.to(x.device)
        sigma = self.sigma.to(x.device)

        z = (x - mu) / sigma

        log_norm = (
            gammaln((nu + 1) / 2)
            - gammaln(nu / 2)
            - 0.5 * torch.log(nu * math.pi)
            - torch.log(sigma)
        )

        log_kernel = -(nu + 1) / 2 * torch.log(1 + z ** 2 / nu)

        log_p = log_norm + log_kernel
        return log_p.sum(dim=-1)

    def sample(self, num_samples: int) -> Tensor:
        return self.rsample(num_samples)

    def rsample(self, num_samples: int) -> Tensor:
        """
        Reparameterized sampling from Student-t.

        Uses the representation: X = mu + sigma * Z / sqrt(V/nu)
        where Z ~ N(0,1) and V ~ Chi2(nu).

        Chi2(nu) is sampled via Gamma(nu/2, 1) * 2, reparameterized
        using the Gamma rsample.
        """
        device = self.mu.device
        dtype = self.mu.dtype

        nu = self.nu
        mu = self.mu
        sigma = self.sigma

        # Sample Z ~ N(0,1)
        Z = torch.randn(num_samples, self.dim, device=device, dtype=dtype)

        # Sample V ~ Chi2(nu) = Gamma(nu/2, 1) * 2
        # Use torch Gamma distribution for reparameterized sampling
        gamma_dist = torch.distributions.Gamma(nu / 2, torch.ones_like(nu))
        V = gamma_dist.rsample((num_samples,)) * 2

        # Student-t sample
        samples = mu + sigma * Z / torch.sqrt(V / nu + 1e-8)

        return samples

    def get_params(self) -> dict:
        return {
            "nu": self.nu.mean().item(),
            "mu": self.mu.mean().item(),
            "sigma": self.sigma.mean().item(),
        }

    def mean(self) -> Tensor:
        """Mean exists for nu > 1."""
        return self.mu

    def variance(self) -> Tensor:
        """Variance = sigma^2 * nu / (nu - 2) for nu > 2."""
        nu = self.nu
        return self.sigma ** 2 * nu / (nu - 2 + 1e-8)




def _gamma_function(x: Tensor) -> Tensor:
    """
    Compute gamma function using exp(gammaln(x)).

    Pure PyTorch implementation for differentiability.
    """
    return torch.exp(gammaln(x))



class CGMY(LevyDistribution):
    """
    CGMY distribution parameterized by (C, G, M, Y).

    The CGMY characteristic function is:
        φ(u) = exp(CΓ(-Y)[(M-iu)^Y - M^Y + (G+iu)^Y - G^Y])

    where Γ is the gamma function.

    Since there's no closed-form PDF, we use:
    1. FFT-based density evaluation for log_prob
    2. Acceptance-rejection sampling with VG proposal

    Parameters:
        C: Jump intensity (C > 0)
        G: Negative jump decay rate (G ≥ 0)
        M: Positive jump decay rate (M ≥ 0)
        Y: Tail/activity parameter (Y < 2)
    """

    def __init__(
        self,
        dim: int = 1,
        C: float = 1.0,
        G: float = 5.0,
        M: float = 5.0,
        Y: float = 0.5,
        learnable: bool = True,
        fft_n: int = 2 ** 12,
        fft_eta: float = 0.1,
    ):
        """
        Initialize CGMY distribution.

        Args:
            dim: Dimensionality (independent CGMY marginals)
            C: Jump intensity (> 0)
            G: Negative jump decay (≥ 0)
            M: Positive jump decay (≥ 0)
            Y: Tail parameter (< 2)
            learnable: Whether parameters are learnable
            fft_n: Number of FFT points for density evaluation
            fft_eta: FFT grid spacing
        """
        super().__init__(dim=dim, learnable=learnable)

        assert C > 0, "C must be positive"
        assert G >= 0, "G must be non-negative"
        assert M >= 0, "M must be non-negative"
        assert Y < 2, "Y must be less than 2"

        self._init_C = C
        self._init_G = G
        self._init_M = M
        self._init_Y = Y

        self.fft_n = fft_n
        self.fft_eta = fft_eta

        if learnable:
            self._log_C = nn.Parameter(
                torch.tensor([math.log(C)] * dim, dtype=torch.float32)
            )
            self._log_G = nn.Parameter(
                torch.tensor([math.log(G + 1e-4)] * dim, dtype=torch.float32)
            )
            self._log_M = nn.Parameter(
                torch.tensor([math.log(M + 1e-4)] * dim, dtype=torch.float32)
            )
            # Y constrained to (-∞, 2) using sigmoid transform
            self._Y_raw = nn.Parameter(
                torch.tensor([self._y_to_raw(Y)] * dim, dtype=torch.float32)
            )
        else:
            self.register_buffer(
                "_log_C", torch.tensor([math.log(C)] * dim, dtype=torch.float32)
            )
            self.register_buffer(
                "_log_G", torch.tensor([math.log(G + 1e-4)] * dim, dtype=torch.float32)
            )
            self.register_buffer(
                "_log_M", torch.tensor([math.log(M + 1e-4)] * dim, dtype=torch.float32)
            )
            self.register_buffer(
                "_Y_raw", torch.tensor([self._y_to_raw(Y)] * dim, dtype=torch.float32)
            )

        # Cache for FFT-based density
        self._pdf_cache = None
        self._pdf_grid = None

    @staticmethod
    def _y_to_raw(y: float) -> float:
        """Transform Y ∈ (-∞, 2) to unconstrained space."""
        # Use logit transform: y_raw = log((2-y)/(y+2)) for y ∈ (-2, 2)
        # Extended version for full range
        return math.log(2 - y + 1e-4)

    @staticmethod
    def _raw_to_y(y_raw: Tensor) -> Tensor:
        """Transform from unconstrained to Y ∈ (-∞, 2)."""
        return 2 - torch.exp(y_raw)

    @property
    def C(self) -> Tensor:
        """Jump intensity parameter (positive)."""
        return torch.exp(self._log_C)

    @property
    def G(self) -> Tensor:
        """Negative jump decay rate (non-negative)."""
        return torch.exp(self._log_G)

    @property
    def M(self) -> Tensor:
        """Positive jump decay rate (non-negative)."""
        return torch.exp(self._log_M)

    @property
    def Y(self) -> Tensor:
        """Tail/activity parameter (< 2)."""
        return self._raw_to_y(self._Y_raw)

    @property
    def tail_index(self) -> float:
        """
        CGMY has power-law tails with index determined by Y.

        For CGMY, the tails behave as:
            P(X > x) ~ x^(-Y-1) * exp(-Mx) for x → +∞
            P(X < -x) ~ x^(-Y-1) * exp(-Gx) for x → +∞

        The effective tail index is Y + 1 for the power-law part,
        but the exponential decay dominates for large G, M.
        When G, M are small, the effective index approaches Y + 1.
        """
        return self.Y.mean().item() + 1

    @property
    def is_symmetric(self) -> bool:
        """Check if distribution is symmetric (G ≈ M)."""
        return torch.allclose(self.G, self.M, atol=1e-6)

    def characteristic_function(self, u: Tensor) -> Tensor:
        """
        Compute CGMY characteristic function.

        φ(u) = exp(CΓ(-Y)[(M-iu)^Y - M^Y + (G+iu)^Y - G^Y])

        For Y = 0 (VG case), use limit:
        φ(u) = exp(C[log(M-iu) - log(M) + log(G+iu) - log(G)])

        Fully differentiable and GPU-compatible using pure PyTorch.

        Args:
            u: Frequency tensor (real)

        Returns:
            Complex tensor of CF values
        """
        C = self.C.to(u.device)
        G = self.G.to(u.device)
        M = self.M.to(u.device)
        Y = self.Y.to(u.device)

        # Complex arithmetic
        # (M - iu)^Y and (G + iu)^Y
        M_minus_iu = torch.complex(M.expand_as(u), -u)
        G_plus_iu = torch.complex(G.expand_as(u), u)

        # Handle Y ≈ 0 (VG limit) - use smooth blending
        abs_Y = torch.abs(Y)
        is_vg = abs_Y < 0.01  # Soft threshold

        # VG limit: Γ(-Y) * [...] → [log(M-iu) - log(M) + log(G+iu) - log(G)]
        # as Y → 0 using L'Hôpital: Γ(-Y) * Y → -1
        log_exp_vg = C * (
            torch.log(M_minus_iu) - torch.log(M.to(torch.complex64))
            + torch.log(G_plus_iu) - torch.log(G.to(torch.complex64))
        )

        # General case: Γ(-Y) for Y not integer
        # Use reflection formula: Γ(-Y) = -π / (Y * Γ(Y) * sin(πY))
        # For Y ∈ (0, 2), -Y ∈ (-2, 0), use gammaln carefully
        # Γ(-Y) = π / (Γ(1+Y) * sin(πY) * (-Y))  (reflection formula variant)
        neg_Y = -Y
        # For Y near integers, this can be problematic
        # Use: Γ(-Y) = Γ(1-Y) / (-Y) for Y > 0
        # And Γ(-Y) = Γ(-Y) directly using reflection for negative Y

        # Safe computation via reflection: Γ(z)Γ(1-z) = π/sin(πz)
        # So Γ(-Y) = π / (sin(-πY) * Γ(1+Y)) = -π / (sin(πY) * Γ(1+Y))
        sin_pi_Y = torch.sin(math.pi * Y)
        gamma_1_plus_Y = _gamma_function(1 + Y)

        # Avoid division by zero when sin(πY) ≈ 0 (Y near integer)
        sin_pi_Y_safe = torch.where(
            torch.abs(sin_pi_Y) < 1e-6,
            torch.ones_like(sin_pi_Y) * 1e-6 * torch.sign(sin_pi_Y + 1e-10),
            sin_pi_Y
        )
        gamma_neg_Y = -math.pi / (sin_pi_Y_safe * gamma_1_plus_Y)

        M_Y = torch.pow(M.to(torch.complex64), Y.to(torch.complex64))
        G_Y = torch.pow(G.to(torch.complex64), Y.to(torch.complex64))
        M_iu_Y = torch.pow(M_minus_iu, Y.unsqueeze(0).to(torch.complex64))
        G_iu_Y = torch.pow(G_plus_iu, Y.unsqueeze(0).to(torch.complex64))

        exponent_general = C.to(torch.complex64) * gamma_neg_Y.to(torch.complex64) * (
            M_iu_Y - M_Y + G_iu_Y - G_Y
        )

        # Blend between VG and general case
        blend_weight = torch.sigmoid((abs_Y - 0.01) * 100)  # Smooth transition
        exponent = blend_weight * exponent_general + (1 - blend_weight) * log_exp_vg

        cf = torch.exp(exponent)

        return cf

    def _compute_pdf_fft_torch(self, x: Tensor, d: int = 0) -> Tensor:
        """
        Compute PDF using FFT-based Fourier inversion with pure PyTorch.

        Uses the Carr-Madan FFT method with dampening.
        Fully differentiable and GPU-compatible.

        Args:
            x: Input points at which to evaluate PDF
            d: Dimension index

        Returns:
            PDF values at x
        """
        device = x.device
        dtype = x.dtype
        N = self.fft_n
        eta = self.fft_eta

        # Get parameters for this dimension
        C = self.C[d:d+1].to(device)
        G = self.G[d:d+1].to(device)
        M = self.M[d:d+1].to(device)
        Y = self.Y[d:d+1].to(device)

        # FFT grid parameters
        lambda_ = 2 * math.pi / (N * eta)
        b = N * lambda_ / 2

        # Frequency grid
        u = torch.arange(N, device=device, dtype=dtype) * eta

        # Dampening parameter for numerical stability
        alpha = 1.5

        # Compute characteristic function at dampened frequencies: u - iα
        # We need CF(u - iα) = CF evaluated at complex frequency
        # For CGMY: φ(v) where v = u - iα is complex

        # Direct computation of dampened CF
        # φ(u - iα) = exp(CΓ(-Y)[(M - i(u-iα))^Y - M^Y + (G + i(u-iα))^Y - G^Y])
        # = exp(CΓ(-Y)[(M - iu - α)^Y - M^Y + (G + iu + α)^Y - G^Y])

        # M - iu - α and G + iu + α as complex numbers
        M_complex = torch.complex(M - alpha, -u)  # M - α - iu
        G_complex = torch.complex(G + alpha, u)   # G + α + iu

        abs_Y = torch.abs(Y)

        # Compute Γ(-Y) using reflection formula
        sin_pi_Y = torch.sin(math.pi * Y)
        gamma_1_plus_Y = _gamma_function(1 + Y)
        sin_pi_Y_safe = torch.where(
            torch.abs(sin_pi_Y) < 1e-6,
            torch.ones_like(sin_pi_Y) * 1e-6 * torch.sign(sin_pi_Y + 1e-10),
            sin_pi_Y
        )
        gamma_neg_Y = -math.pi / (sin_pi_Y_safe * gamma_1_plus_Y)

        # Compute CF values
        M_Y = torch.pow(M.to(torch.complex64), Y.to(torch.complex64))
        G_Y = torch.pow(G.to(torch.complex64), Y.to(torch.complex64))
        M_c_Y = torch.pow(M_complex.to(torch.complex64), Y.to(torch.complex64))
        G_c_Y = torch.pow(G_complex.to(torch.complex64), Y.to(torch.complex64))

        # VG limit for Y ≈ 0
        cf_vg = torch.exp(C * (
            torch.log(M_complex.to(torch.complex64)) - torch.log((M - alpha).to(torch.complex64))
            + torch.log(G_complex.to(torch.complex64)) - torch.log((G + alpha).to(torch.complex64))
        ))

        # General case
        cf_general = torch.exp(
            C.to(torch.complex64) * gamma_neg_Y.to(torch.complex64) * (
                M_c_Y - M_Y + G_c_Y - G_Y
            )
        )

        # Blend
        blend_weight = torch.sigmoid((abs_Y - 0.01) * 100)
        cf_vals = blend_weight * cf_general + (1 - blend_weight) * cf_vg

        # Simpson's rule weights
        indices = torch.arange(N, device=device)
        simpson = torch.ones(N, device=device, dtype=dtype)
        simpson[1::2] = 4.0  # Odd indices
        simpson[2::2] = 2.0  # Even indices (except first and last)
        simpson[0] = 1.0
        simpson[-1] = 1.0 if N % 2 == 1 else (4.0 if N % 2 == 0 else 2.0)
        simpson = simpson / 3.0

        # Phase shift for centering
        phase = torch.exp(1j * u * b)

        # FFT integrand
        integrand = cf_vals.squeeze() * phase * eta * simpson.to(torch.complex64)

        # Apply FFT
        fft_result = torch.fft.fft(integrand)
        fft_result = torch.fft.fftshift(fft_result)

        # Output grid
        k = -b + lambda_ * torch.arange(N, device=device, dtype=dtype)

        # Recover PDF from dampened transform
        pdf_grid = torch.exp(-alpha * k) * fft_result.real / math.pi

        # Ensure non-negative
        pdf_grid = torch.clamp(pdf_grid, min=1e-10)

        # Interpolate to requested x values
        # Use linear interpolation
        x_clamped = torch.clamp(x, min=k[0], max=k[-1])

        # Find indices for interpolation
        dx = k[1] - k[0]
        idx_float = (x_clamped - k[0]) / dx
        idx_low = torch.clamp(idx_float.long(), 0, N - 2)
        idx_high = idx_low + 1

        # Linear interpolation weights
        weight_high = idx_float - idx_low.float()
        weight_low = 1 - weight_high

        # Interpolate
        pdf_interp = weight_low * pdf_grid[idx_low] + weight_high * pdf_grid[idx_high]

        return torch.clamp(pdf_interp, min=1e-10)

    def log_prob(self, x: Tensor) -> Tensor:
        """
        Compute log probability using FFT-based density evaluation.

        Fully differentiable and GPU-compatible using pure PyTorch FFT.

        Args:
            x: Input tensor of shape (batch_size, dim) or (batch_size,)

        Returns:
            Log probabilities of shape (batch_size,)
        """
        if x.dim() == 1:
            x = x.unsqueeze(-1)

        batch_size = x.shape[0]
        log_probs = torch.zeros(batch_size, self.dim, device=x.device, dtype=x.dtype)

        for d in range(self.dim):
            # Compute PDF using differentiable FFT
            pdf_vals = self._compute_pdf_fft_torch(x[:, d], d)
            log_probs[:, d] = torch.log(pdf_vals)

        return log_probs.sum(dim=-1)

    def sample(self, num_samples: int) -> Tensor:
        """Sample from CGMY distribution."""
        return self.rsample(num_samples)

    def rsample(self, num_samples: int) -> Tensor:
        """
        Reparameterized sampling from CGMY using series representation.

        For Y < 0 (finite activity): Use compound Poisson representation
        For Y ≥ 0 (infinite activity): Use shot noise series representation

        This is an approximate method; for high accuracy, use acceptance-rejection
        with a carefully chosen proposal.
        """
        device = self.C.device
        dtype = self.C.dtype

        samples = torch.zeros(num_samples, self.dim, device=device, dtype=dtype)

        for d in range(self.dim):
            C = self.C[d]
            G = self.G[d]
            M = self.M[d]
            Y = self.Y[d]

            if Y < 0:
                # Finite activity: compound Poisson
                samples[:, d] = self._sample_compound_poisson(
                    num_samples, C, G, M, Y
                )
            else:
                # Infinite activity: use series representation
                # Approximate with truncated series
                samples[:, d] = self._sample_series_representation(
                    num_samples, C, G, M, Y
                )

        return samples

    def _sample_compound_poisson(
        self, num_samples: int, C: Tensor, G: Tensor, M: Tensor, Y: Tensor
    ) -> Tensor:
        """
        Sample CGMY with Y < 0 using compound Poisson representation.

        For Y < 0, the process has finite activity and can be written as
        a compound Poisson process.

        Pure PyTorch implementation for GPU compatibility.
        """
        device = C.device
        dtype = C.dtype

        # For Y < 0, total intensity is finite
        # λ = C * Γ(-Y) * (G^Y + M^Y)
        gamma_neg_Y = _gamma_function(-Y)
        intensity = C * gamma_neg_Y * (torch.pow(G, Y) + torch.pow(M, Y))
        intensity_val = intensity.item()

        # Maximum number of jumps to simulate (heuristic upper bound)
        max_jumps = int(intensity_val * 3 + 10)

        samples = torch.zeros(num_samples, device=device, dtype=dtype)

        # Vectorized Poisson sampling
        poisson = torch.distributions.Poisson(intensity)
        n_jumps_all = poisson.sample((num_samples,)).long()

        # Probability of positive jump
        prob_pos = torch.pow(M, Y) / (torch.pow(G, Y) + torch.pow(M, Y) + 1e-10)

        for i in range(num_samples):
            n_jumps = min(n_jumps_all[i].item(), max_jumps)

            if n_jumps > 0:
                # Sample jump directions
                directions = 2 * (torch.rand(n_jumps, device=device) < prob_pos).float() - 1

                # Sample jump sizes from tempered stable using rejection
                # Simplified: use exponential approximation
                jump_sizes = torch.zeros(n_jumps, device=device, dtype=dtype)
                for j in range(n_jumps):
                    lam = M if directions[j] > 0 else G
                    jump_sizes[j] = self._sample_tempered_stable_torch(Y, lam)

                samples[i] = (directions * jump_sizes).sum()

        return samples

    def _sample_tempered_stable_torch(self, Y: Tensor, lam: Tensor) -> Tensor:
        """Sample single value from tempered stable distribution using PyTorch."""
        device = Y.device
        dtype = Y.dtype

        # Use acceptance-rejection with exponential proposal
        max_iter = 1000
        Y_val = Y.item()
        lam_val = lam.item()

        for _ in range(max_iter):
            # Propose from Exp(lam)
            x = torch.distributions.Exponential(lam).sample()
            # Accept with probability proportional to x^(-1-Y)
            accept_prob = torch.clamp(torch.pow(x + 1e-10, -1 - Y_val), max=1.0)
            if torch.rand(1, device=device) < accept_prob:
                return x

        # Fallback
        return torch.distributions.Exponential(lam).sample()

    def _sample_series_representation(
        self, num_samples: int, C: Tensor, G: Tensor, M: Tensor, Y: Tensor
    ) -> Tensor:
        """
        Sample CGMY with Y >= 0 using truncated series representation.

        Uses the shot noise series representation of Lévy processes.
        Pure PyTorch implementation for GPU compatibility.
        """
        device = C.device
        dtype = C.dtype

        # Truncation level
        epsilon = 1e-4
        T = 1.0  # Time horizon

        # Number of terms in series (heuristic)
        n_terms = max(1000, int(C.item() * T / epsilon))

        samples = torch.zeros(num_samples, device=device, dtype=dtype)

        # Probability of positive jump
        prob_pos = torch.pow(M, Y) / (torch.pow(G, Y) + torch.pow(M, Y) + 1e-10)

        for i in range(num_samples):
            # Generate Poisson epochs
            exp_dist = torch.distributions.Exponential(C)
            inter_arrivals = exp_dist.sample((n_terms,))
            epochs = torch.cumsum(inter_arrivals, dim=0)

            # Generate jump signs
            signs = 2 * (torch.rand(n_terms, device=device) < prob_pos).float() - 1

            # Generate jump sizes
            jump_sizes = torch.zeros(n_terms, device=device, dtype=dtype)
            Y_val = Y.item()

            for j in range(n_terms):
                lam = M if signs[j] > 0 else G
                u = epochs[j]

                # Simplified: x ≈ (u / C)^(1/Y) for small jumps
                if Y_val > 0.01:
                    jump_sizes[j] = torch.pow(u / (C + 1e-10), 1.0 / Y_val)
                else:
                    jump_sizes[j] = torch.distributions.Exponential(lam).sample()

                # Apply tempering rejection
                if torch.rand(1, device=device) > torch.exp(-lam * jump_sizes[j]):
                    jump_sizes[j] = 0

            # Truncate small jumps and sum
            valid = jump_sizes > epsilon
            samples[i] = (signs[valid] * jump_sizes[valid]).sum()

        return samples

    def get_params(self) -> dict:
        """Return dictionary of distribution parameters."""
        return {
            "C": self.C.mean().item(),
            "G": self.G.mean().item(),
            "M": self.M.mean().item(),
            "Y": self.Y.mean().item(),
        }

    def reset_parameters(self):
        """Reset parameters to initial values."""
        with torch.no_grad():
            self._log_C.fill_(math.log(self._init_C))
            self._log_G.fill_(math.log(self._init_G + 1e-4))
            self._log_M.fill_(math.log(self._init_M + 1e-4))
            self._Y_raw.fill_(self._y_to_raw(self._init_Y))


def _log_bessel_k1(z: Tensor) -> Tensor:
    """
    Compute log of modified Bessel function K_1(z) using pure PyTorch.

    Uses asymptotic expansion for large z and polynomial approximation for small z.
    Fully differentiable and GPU-compatible.

    For K_1(z):
        Large z: K_1(z) ~ sqrt(pi/(2z)) * exp(-z) * (1 + 3/(8z) + ...)
        Small z: K_1(z) ~ 1/z + z/2 * (log(z/2) + gamma - 1/2)

    Args:
        z: Argument, any shape

    Returns:
        log K_1(z), same shape as z
    """
    z = torch.clamp(z, min=1e-8)

    # Asymptotic expansion for large z:
    # log K_1(z) ≈ 0.5*log(pi/2) - 0.5*log(z) - z + log(1 + 3/(8z) + ...)
    log_leading = 0.5 * math.log(math.pi / 2) - 0.5 * torch.log(z) - z

    # Correction terms for K_1: mu = 4*1^2 = 4
    # term1 = (4-1)/(8z) = 3/(8z)
    # term2 = (4-1)(4-9)/(128z^2) = 3*(-5)/(128z^2) = -15/(128z^2)
    # term3 = (4-1)(4-9)(4-25)/(3072z^3) = 3*(-5)*(-21)/(3072z^3) = 315/(3072z^3)
    term1 = 3 / (8 * z)
    term2 = -15 / (128 * z ** 2)
    term3 = 315 / (3072 * z ** 3)

    correction_sum = 1 + term1 + term2 + term3
    correction_sum = torch.clamp(correction_sum, min=1e-10)
    log_correction = torch.log(correction_sum)

    log_k_asymptotic = log_leading + log_correction

    # For small z, K_1(z) ≈ 1/z (dominant term)
    # log K_1(z) ≈ -log(z) for small z
    log_k_small = -torch.log(z)

    # Blend between approximations
    threshold = 2.0
    blend_weight = torch.sigmoid((z - threshold) * 3)

    log_k = blend_weight * log_k_asymptotic + (1 - blend_weight) * log_k_small

    # Clamp final result
    log_k = torch.clamp(log_k, min=-100, max=100)

    return log_k


class NormalInverseGaussian(LevyDistribution):
    """
    Normal-Inverse Gaussian distribution parameterized by (α, β, μ, δ).

    The NIG density is:
        p(x) = (αδ/π) * exp(δγ + β(x-μ)) * K_1(α√(δ² + (x-μ)²)) / √(δ² + (x-μ)²)

    where γ = √(α² - β²) and K_1 is modified Bessel function of second kind.

    Parameters:
        α (alpha): Tail heaviness parameter (α > 0, larger = lighter tails)
        β (beta): Skewness parameter (|β| < α, β < 0 left skew, β > 0 right skew)
        μ (mu): Location parameter
        δ (delta): Scale parameter (δ > 0)

    Constraints:
        α > 0, δ > 0, |β| < α

    Special cases:
        - β = 0: Symmetric NIG
        - α → ∞: Approaches Gaussian
    """

    def __init__(
        self,
        dim: int = 1,
        alpha: float = 1.0,
        beta: float = 0.0,
        mu: float = 0.0,
        delta: float = 1.0,
        learnable: bool = True,
    ):
        """
        Initialize NIG distribution.

        Args:
            dim: Dimensionality (independent NIG marginals)
            alpha: Tail heaviness (> 0, larger = lighter tails)
            beta: Skewness (-alpha < beta < alpha)
            mu: Location
            delta: Scale (> 0)
            learnable: Whether parameters are learnable
        """
        super().__init__(dim=dim, learnable=learnable)

        # Validate constraints
        assert alpha > 0, "alpha must be positive"
        assert delta > 0, "delta must be positive"
        assert abs(beta) < alpha, "|beta| must be less than alpha"

        self._init_alpha = alpha
        self._init_beta = beta
        self._init_mu = mu
        self._init_delta = delta

        if learnable:
            # Use softplus for positivity constraints
            # alpha is stored as log(alpha) for unconstrained optimization
            self._log_alpha = nn.Parameter(
                torch.tensor([math.log(alpha)] * dim, dtype=torch.float32)
            )
            # beta is stored as tanh^{-1}(beta/alpha) to enforce |beta| < alpha
            # We'll handle this constraint more carefully
            self._beta_raw = nn.Parameter(
                torch.tensor([beta / alpha] * dim, dtype=torch.float32)
            )
            self._mu = nn.Parameter(torch.tensor([mu] * dim, dtype=torch.float32))
            self._log_delta = nn.Parameter(
                torch.tensor([math.log(delta)] * dim, dtype=torch.float32)
            )
        else:
            self.register_buffer(
                "_log_alpha", torch.tensor([math.log(alpha)] * dim, dtype=torch.float32)
            )
            self.register_buffer(
                "_beta_raw", torch.tensor([beta / alpha] * dim, dtype=torch.float32)
            )
            self.register_buffer("_mu", torch.tensor([mu] * dim, dtype=torch.float32))
            self.register_buffer(
                "_log_delta", torch.tensor([math.log(delta)] * dim, dtype=torch.float32)
            )

    @property
    def alpha(self) -> Tensor:
        """Tail heaviness parameter (positive)."""
        return torch.exp(self._log_alpha)

    @property
    def beta(self) -> Tensor:
        """Skewness parameter (constrained to |β| < α)."""
        # Use tanh to constrain to (-1, 1), then scale by alpha
        return self.alpha * torch.tanh(self._beta_raw) * 0.99  # 0.99 for numerical safety

    @property
    def mu(self) -> Tensor:
        """Location parameter."""
        return self._mu

    @property
    def delta(self) -> Tensor:
        """Scale parameter (positive)."""
        return torch.exp(self._log_delta)

    @property
    def gamma(self) -> Tensor:
        """Derived parameter γ = √(α² - β²)."""
        return torch.sqrt(self.alpha ** 2 - self.beta ** 2 + 1e-8)

    @property
    def tail_index(self) -> float:
        """
        NIG has semi-heavy tails with effective tail index around 1.

        The NIG tails decay as:
            p(x) ~ |x|^(-3/2) * exp(-α|x|) as |x| → ∞

        This is semi-heavy: polynomial prefactor with exponential decay.
        For practical purposes, the effective tail index is approximately 1-2.
        """
        # Effective tail index based on the polynomial prefactor
        return 1.5

    @property
    def is_symmetric(self) -> bool:
        """Check if distribution is symmetric (beta ≈ 0)."""
        return torch.allclose(self.beta, torch.zeros_like(self.beta), atol=1e-6)

    def log_prob(self, x: Tensor) -> Tensor:
        """
        Compute log probability density of NIG distribution.

        log p(x) = log(α) + log(δ) - log(π) + δγ + β(x-μ)
                   + log(K_1(αq)) - log(q)

        where q = √(δ² + (x-μ)²) and γ = √(α² - β²)

        Fully differentiable and GPU-compatible using pure PyTorch Bessel K
        approximation.

        Args:
            x: Input tensor of shape (batch_size, dim) or (batch_size,)

        Returns:
            Log probabilities of shape (batch_size,)
        """
        if x.dim() == 1:
            x = x.unsqueeze(-1)

        # Get parameters - keep on same device as x
        alpha = self.alpha.to(x.device)
        beta = self.beta.to(x.device)
        mu = self.mu.to(x.device)
        delta = self.delta.to(x.device)
        gamma = torch.sqrt(alpha ** 2 - beta ** 2 + 1e-8)

        # q = √(δ² + (x-μ)²)
        x_centered = x - mu
        q = torch.sqrt(delta ** 2 + x_centered ** 2 + 1e-8)

        # Bessel argument
        bessel_arg = alpha * q

        # Log normalization and exponential terms
        log_norm = torch.log(alpha) + torch.log(delta) - math.log(math.pi)
        exp_term = delta * gamma + beta * x_centered

        # Log Bessel function using pure PyTorch (differentiable, GPU-compatible)
        log_bessel = _log_bessel_k1(bessel_arg)

        # Combine: log(K_1(αq)/q) = log(K_1(αq)) - log(q)
        log_p = log_norm + exp_term + log_bessel - torch.log(q)

        # Sum over dimensions
        return log_p.sum(dim=-1)

    def sample(self, num_samples: int) -> Tensor:
        """Sample from NIG distribution."""
        return self.rsample(num_samples)

    def rsample(self, num_samples: int) -> Tensor:
        """
        Reparameterized sampling from NIG distribution.

        NIG can be sampled as a normal variance-mean mixture:
            X = μ + βV + √V * Z
        where V ~ InverseGaussian(δ/γ, δ²) and Z ~ N(0, 1)

        The InverseGaussian is sampled using the method of Michael et al. (1976).
        """
        device = self.mu.device
        dtype = self.mu.dtype

        alpha = self.alpha
        beta = self.beta
        mu = self.mu
        delta = self.delta
        gamma = self.gamma

        # Inverse Gaussian parameters
        # V ~ IG(μ_ig, λ_ig) where μ_ig = δ/γ, λ_ig = δ²
        ig_mu = delta / gamma
        ig_lambda = delta ** 2

        # Sample Inverse Gaussian using Michael et al. (1976) algorithm
        # This is reparameterizable through implicit differentiation
        V = self._sample_inverse_gaussian(num_samples, ig_mu, ig_lambda)

        # Sample standard normal
        Z = torch.randn(num_samples, self.dim, device=device, dtype=dtype)

        # NIG sample: X = μ + βV + √V * Z
        samples = mu + beta * V + torch.sqrt(V + 1e-8) * Z

        return samples

    def _sample_inverse_gaussian(
        self, num_samples: int, mu: Tensor, lam: Tensor
    ) -> Tensor:
        """
        Sample from Inverse Gaussian distribution using Michael et al. (1976).

        IG(μ, λ) has density:
            p(x) = √(λ/(2πx³)) * exp(-λ(x-μ)²/(2μ²x))

        Algorithm:
        1. Sample Y ~ N(0,1)², i.e., chi-squared with 1 df
        2. Compute X = μ + μ²Y/(2λ) - μ/(2λ)√(4μλY + μ²Y²)
        3. Sample U ~ Uniform(0,1)
        4. If U ≤ μ/(μ+X), return X; else return μ²/X
        """
        device = mu.device
        dtype = mu.dtype

        # Sample chi-squared(1) = N(0,1)²
        Z = torch.randn(num_samples, self.dim, device=device, dtype=dtype)
        Y = Z ** 2

        # Compute candidate X
        mu2 = mu ** 2
        term1 = mu + mu2 * Y / (2 * lam)
        term2 = mu / (2 * lam) * torch.sqrt(4 * mu * lam * Y + mu2 * Y ** 2 + 1e-8)
        X1 = term1 - term2

        # Ensure positivity
        X1 = torch.clamp(X1, min=1e-8)

        # Accept/reject
        U = torch.rand(num_samples, self.dim, device=device, dtype=dtype)
        accept = U <= mu / (mu + X1)

        # Alternative value
        X2 = mu2 / X1

        # Select based on acceptance
        V = torch.where(accept, X1, X2)

        return V

    def characteristic_function(self, t: Tensor) -> Tensor:
        """
        Compute characteristic function of NIG distribution.

        φ(t) = exp(iμt + δ(γ - √(α² - (β + it)²)))

        Args:
            t: Frequency tensor

        Returns:
            Complex tensor of CF values
        """
        alpha = self.alpha
        beta = self.beta
        mu = self.mu
        delta = self.delta
        gamma = self.gamma

        # Complex argument: β + it
        beta_plus_it = torch.complex(beta.expand_as(t), t)

        # α² - (β + it)²
        alpha2 = alpha ** 2
        inner = alpha2 - beta_plus_it ** 2

        # √(α² - (β + it)²) - need to handle branch cut
        sqrt_inner = torch.sqrt(inner)

        # Exponent: iμt + δ(γ - √(...))
        i_mu_t = torch.complex(torch.zeros_like(t), mu * t)
        exponent = i_mu_t + delta * (gamma - sqrt_inner)

        return torch.exp(exponent)

    def get_params(self) -> dict:
        """Return dictionary of distribution parameters."""
        return {
            "alpha": self.alpha.mean().item(),
            "beta": self.beta.mean().item(),
            "mu": self.mu.mean().item(),
            "delta": self.delta.mean().item(),
        }

    def mean(self) -> Tensor:
        """Compute mean of NIG distribution: E[X] = μ + δβ/γ"""
        return self.mu + self.delta * self.beta / self.gamma

    def variance(self) -> Tensor:
        """Compute variance of NIG distribution: Var(X) = δα²/γ³"""
        return self.delta * (self.alpha ** 2) / (self.gamma ** 3)

    def skewness(self) -> Tensor:
        """Compute skewness of NIG distribution: 3β / (α√(δγ))"""
        return 3 * self.beta / (self.alpha * torch.sqrt(self.delta * self.gamma + 1e-8))

    def kurtosis(self) -> Tensor:
        """Compute excess kurtosis of NIG: 3(1 + 4β²/α²) / (δγ)"""
        return 3 * (1 + 4 * (self.beta ** 2) / (self.alpha ** 2)) / (self.delta * self.gamma)

    def reset_parameters(self):
        """Reset parameters to initial values."""
        with torch.no_grad():
            self._log_alpha.fill_(math.log(self._init_alpha))
            self._beta_raw.fill_(self._init_beta / self._init_alpha)
            self._mu.fill_(self._init_mu)
            self._log_delta.fill_(math.log(self._init_delta))