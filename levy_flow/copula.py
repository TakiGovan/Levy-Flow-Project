"""
Copula module: build a joint multivariate distribution from independently
fitted univariate marginals (e.g., per-asset VG-flow models).

Theory
------
Sklar's theorem: any joint CDF factors as

    F(x_1,...,x_d) = C(F_1(x_1),...,F_d(x_d))

uniquely, given continuous marginals F_i. In density form:

    f(x_1,...,x_d) = c(u_1,...,u_d) * prod_i f_i(x_i),    u_i = F_i(x_i)

so a joint model = (already-fitted marginals) + (a copula c on [0,1]^d).
Fit via the standard two-stage "Inference Functions for Margins" (IFM)
procedure: fit marginals first (fixed henceforth), transform data to
pseudo-observations U_i = F_i(X_i), then fit a copula family to U by MLE.

Gaussian copula
---------------
    C(u; R) = Phi_R(Phi^-1(u_1), ..., Phi^-1(u_d))
    log c(u; R) = -1/2 log|R| - 1/2 z^T (R^-1 - I) z,   z_i = Phi^-1(u_i)

Zero tail dependence for any |rho| < 1:
    lambda_U = lim_{q->1} P(U_2 > q | U_1 > q) = 0
A Gaussian copula cannot produce joint extreme events more often than its
bulk correlation implies -- the textbook mechanism behind pre-2008 model
failure on portfolio crash risk.

Student-t copula
-----------------
    C(u; R, nu) = t_{R,nu}(t_nu^-1(u_1), ..., t_nu^-1(u_d))

Nonzero tail dependence for any rho > -1:
    lambda_U = 2 * t_{nu+1}(-sqrt((nu+1)(1-rho)/(1+rho)))
Assets can crash together more often than bulk correlation alone implies --
the property that actually matters for portfolio tail risk, and the concrete,
checkable reason a t-copula (or richer) should beat a Gaussian copula here.
"""

import math
import numpy as np
from scipy.stats import norm, t as student_t


def pit_from_samples(x: np.ndarray, ref_samples: np.ndarray) -> np.ndarray:
    """
    Probability integral transform via Monte Carlo: estimate F(x) using a
    large reference sample drawn from the fitted marginal (e.g.
    flow.sample(200_000)), rather than requiring a closed-form CDF. Standard
    and robust for models (like VG-flows) with no simple closed-form CDF.
    """
    ref_sorted = np.sort(ref_samples)
    u = np.searchsorted(ref_sorted, x, side="right") / len(ref_sorted)
    return np.clip(u, 1e-6, 1 - 1e-6)


class GaussianCopula:
    """Gaussian copula: fit correlation matrix R by MLE (closed form: sample
    correlation of the normal-scores z_i = Phi^-1(u_i))."""

    def __init__(self, dim: int):
        self.dim = dim
        self.R = np.eye(dim)

    def fit(self, U: np.ndarray) -> "GaussianCopula":
        U = np.clip(U, 1e-6, 1 - 1e-6)
        Z = norm.ppf(U)
        self.R = np.corrcoef(Z, rowvar=False)
        return self

    def log_prob(self, U: np.ndarray) -> np.ndarray:
        U = np.clip(U, 1e-6, 1 - 1e-6)
        Z = norm.ppf(U)
        Rinv = np.linalg.inv(self.R)
        _, logdet = np.linalg.slogdet(self.R)
        quad = np.einsum("ni,ij,nj->n", Z, Rinv - np.eye(self.dim), Z)
        return -0.5 * logdet - 0.5 * quad

    def sample(self, n: int) -> np.ndarray:
        L = np.linalg.cholesky(self.R)
        Z = np.random.standard_normal((n, self.dim)) @ L.T
        return norm.cdf(Z)

    def tail_dependence(self) -> float:
        return 0.0  # exact, for any |rho| < 1 -- see module docstring


class StudentTCopula:
    """Student-t copula: correlation matrix R plus shared degrees-of-freedom
    nu, fit by profile MLE over a nu grid (R has a fast closed-form-ish
    estimate at each nu via the t-scores' sample correlation)."""

    def __init__(self, dim: int):
        self.dim = dim
        self.R = np.eye(dim)
        self.nu = 5.0

    def _loglik(self, U: np.ndarray, R: np.ndarray, nu: float) -> float:
        d = self.dim
        Tq = student_t.ppf(U, df=nu)
        Rinv = np.linalg.inv(R)
        _, logdet = np.linalg.slogdet(R)
        quad = np.einsum("ni,ij,nj->n", Tq, Rinv, Tq)
        log_c = (
            math.lgamma((nu + d) / 2) + (d - 1) * math.lgamma(nu / 2)
            - d * math.lgamma((nu + 1) / 2)
            - 0.5 * logdet
            - (nu + d) / 2 * np.log1p(quad / nu)
        )
        log_c = log_c - student_t.logpdf(Tq, df=nu).sum(axis=1)
        return float(log_c.sum())

    def fit(self, U: np.ndarray, nu_grid=None) -> "StudentTCopula":
        U = np.clip(U, 1e-6, 1 - 1e-6)
        if nu_grid is None:
            nu_grid = [3, 4, 5, 6, 8, 10, 15, 20, 30, 50]
        best = None
        for nu in nu_grid:
            Tq = student_t.ppf(U, df=nu)
            R = np.corrcoef(Tq, rowvar=False)
            ll = self._loglik(U, R, nu)
            if best is None or ll > best[0]:
                best = (ll, R, nu)
        self.R, self.nu = best[1], best[2]
        return self

    def log_prob(self, U: np.ndarray) -> np.ndarray:
        U = np.clip(U, 1e-6, 1 - 1e-6)
        d = self.dim
        Tq = student_t.ppf(U, df=self.nu)
        Rinv = np.linalg.inv(self.R)
        _, logdet = np.linalg.slogdet(self.R)
        quad = np.einsum("ni,ij,nj->n", Tq, Rinv, Tq)
        log_c = (
            math.lgamma((self.nu + d) / 2) + (d - 1) * math.lgamma(self.nu / 2)
            - d * math.lgamma((self.nu + 1) / 2)
            - 0.5 * logdet
            - (self.nu + d) / 2 * np.log1p(quad / self.nu)
        )
        return log_c - student_t.logpdf(Tq, df=self.nu).sum(axis=1)

    def sample(self, n: int) -> np.ndarray:
        L = np.linalg.cholesky(self.R)
        Z = np.random.standard_normal((n, self.dim)) @ L.T
        chi2 = np.random.chisquare(self.nu, size=n)
        Tq = Z / np.sqrt(chi2 / self.nu)[:, None]
        return student_t.cdf(Tq, df=self.nu)

    def tail_dependence(self) -> float:
        """Pairwise (0,1) tail dependence coefficient -- exact formula."""
        rho = self.R[0, 1]
        arg = -math.sqrt((self.nu + 1) * (1 - rho) / (1 + rho))
        return 2 * student_t.cdf(arg, df=self.nu + 1)


class JointModel:
    """
    Combine per-asset marginal models (anything with .log_prob(x) and
    .sample(n)) with a fitted copula into a joint model.

    log_prob(X) = sum_i log f_i(x_i) + log c(U)   [Sklar's theorem, density form]
    """

    def __init__(self, marginals: list, copula):
        self.marginals = marginals
        self.copula = copula
        self.dim = len(marginals)
        self._ref_samples = None  # cached large samples for PIT

    def _get_ref_samples(self, n_ref: int = 200_000):
        if self._ref_samples is None:
            self._ref_samples = []
            for m in self.marginals:
                s = m.sample(n_ref)
                if hasattr(s, "detach"):
                    s = s.detach().cpu().numpy()
                self._ref_samples.append(np.asarray(s).flatten())
        return self._ref_samples

    def pit(self, X: np.ndarray) -> np.ndarray:
        """X: (N, dim) -> U: (N, dim) pseudo-observations."""
        refs = self._get_ref_samples()
        U = np.zeros_like(X, dtype=float)
        for i in range(self.dim):
            U[:, i] = pit_from_samples(X[:, i], refs[i])
        return U

    def log_prob(self, X: np.ndarray, marginal_logpdf_fn=None) -> np.ndarray:
        """
        marginal_logpdf_fn(i, x_i) -> log f_i(x_i) array. If not provided,
        assumes self.marginals[i].log_prob(x_i) returns exactly that (as a
        numpy array or a tensor with .numpy()).
        """
        U = self.pit(X)
        log_c = self.copula.log_prob(U)
        log_marg = np.zeros(X.shape[0])
        for i, m in enumerate(self.marginals):
            if marginal_logpdf_fn is not None:
                lp = marginal_logpdf_fn(i, X[:, i])
            else:
                import torch
                with torch.no_grad():
                    lp = m.log_prob(torch.tensor(X[:, i], dtype=torch.float32))
                lp = lp.numpy()
            log_marg += lp
        return log_c + log_marg

    def sample(self, n: int) -> np.ndarray:
        """Sample jointly: draw U from the copula, invert each marginal's CDF
        via the same reference-sample quantile trick used for PIT."""
        U = self.copula.sample(n)
        refs = self._get_ref_samples()
        X = np.zeros((n, self.dim))
        for i in range(self.dim):
            X[:, i] = np.quantile(refs[i], U[:, i])
        return X
