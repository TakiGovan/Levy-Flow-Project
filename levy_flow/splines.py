"""
Rational-quadratic spline coupling transforms (Durkan, Bekasov, Murray & Papamakarios,
"Neural Spline Flows", NeurIPS 2019), with linear ("identity-like") tails outside
[-tail_bound, tail_bound] so that the transform is the identity map beyond the
support of the spline. This lets a spline-based flow preserve whatever tail
behaviour the base distribution already has (here: Variance Gamma tails), since
mass that lands beyond tail_bound is passed through unchanged rather than being
forced back into a bounded interval.

This is a from-scratch reimplementation (no zuko / nflows dependency).
"""

import torch
import torch.nn.functional as F

DEFAULT_MIN_BIN_WIDTH = 1e-3
DEFAULT_MIN_BIN_HEIGHT = 1e-3
DEFAULT_MIN_DERIVATIVE = 1e-3


# def searchsorted(bin_locations: torch.Tensor, inputs: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
#     """Find bin index of each element of `inputs` within `bin_locations` (last dim)."""
#     bin_locations = bin_locations.clone()
#     bin_locations[..., -1] += eps
#     return torch.sum(inputs[..., None] >= bin_locations, dim=-1) - 1

def searchsorted(bin_locations: torch.Tensor, inputs: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Find bin index of each element of `inputs` within `bin_locations` (last dim).
    Clamps to valid range [0, num_bins - 1].
    """
    bin_locations = bin_locations.clone()
    bin_locations[..., -1] += eps
    
    # Use torch.bucketize for better handling
    # But keep the original logic with clamping
    idx = torch.sum(inputs[..., None] >= bin_locations, dim=-1) - 1
    idx = torch.clamp(idx, min=0, max=bin_locations.shape[-1] - 2)
    return idx


def unconstrained_rational_quadratic_spline(
    inputs: torch.Tensor,
    unnormalized_widths: torch.Tensor,
    unnormalized_heights: torch.Tensor,
    unnormalized_derivatives: torch.Tensor,
    inverse: bool = False,
    tail_bound: float = 5.0,
    min_bin_width: float = DEFAULT_MIN_BIN_WIDTH,
    min_bin_height: float = DEFAULT_MIN_BIN_HEIGHT,
    min_derivative: float = DEFAULT_MIN_DERIVATIVE,
):
    """
    RQS transform on the whole real line: identity outside [-tail_bound, tail_bound],
    a monotonic rational-quadratic spline inside. Boundary derivatives of the spline
    are pinned to 1 so the transform (and its derivative) is continuous at the seam.

    inputs: (...,)
    unnormalized_{widths,heights}: (..., num_bins)
    unnormalized_derivatives: (..., num_bins - 1)   [internal knot derivatives only;
                                                       the two boundary derivatives are fixed to 1]
    Returns (outputs, logabsdet), both shaped like `inputs`.
    """
    inside_interval_mask = (inputs >= -tail_bound) & (inputs <= tail_bound)
    outside_interval_mask = ~inside_interval_mask

    outputs = torch.zeros_like(inputs)
    logabsdet = torch.zeros_like(inputs)

    # Identity tails: y = x, log|dy/dx| = 0
    outputs[outside_interval_mask] = inputs[outside_interval_mask]
    logabsdet[outside_interval_mask] = 0.0

    # Pad derivatives with fixed boundary slopes of 1 (in "pre-softplus" space this is
    # the constant that makes softplus(c) + min_derivative == 1).
    constant = float(torch.log(torch.exp(torch.tensor(1.0 - min_derivative)) - 1.0))
    unnormalized_derivatives = F.pad(unnormalized_derivatives, pad=(1, 1))
    unnormalized_derivatives[..., 0] = constant
    unnormalized_derivatives[..., -1] = constant

    if inside_interval_mask.any():
        out_in, logabsdet_in = rational_quadratic_spline(
            inputs=inputs[inside_interval_mask],
            unnormalized_widths=unnormalized_widths[inside_interval_mask, :],
            unnormalized_heights=unnormalized_heights[inside_interval_mask, :],
            unnormalized_derivatives=unnormalized_derivatives[inside_interval_mask, :],
            inverse=inverse,
            left=-tail_bound, right=tail_bound, bottom=-tail_bound, top=tail_bound,
            min_bin_width=min_bin_width, min_bin_height=min_bin_height, min_derivative=min_derivative,
        )
        outputs[inside_interval_mask] = out_in
        logabsdet[inside_interval_mask] = logabsdet_in

    return outputs, logabsdet


def rational_quadratic_spline(
    inputs, unnormalized_widths, unnormalized_heights, unnormalized_derivatives,
    inverse=False, left=0.0, right=1.0, bottom=0.0, top=1.0,
    min_bin_width=DEFAULT_MIN_BIN_WIDTH, min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
    min_derivative=DEFAULT_MIN_DERIVATIVE,
):
    num_bins = unnormalized_widths.shape[-1]
    if min_bin_width * num_bins > 1.0:
        raise ValueError("min_bin_width * num_bins must be <= 1")
    if min_bin_height * num_bins > 1.0:
        raise ValueError("min_bin_height * num_bins must be <= 1")

    widths = F.softmax(unnormalized_widths, dim=-1)
    widths = min_bin_width + (1 - min_bin_width * num_bins) * widths
    cumwidths = torch.cumsum(widths, dim=-1)
    cumwidths = F.pad(cumwidths, pad=(1, 0), value=0.0)
    cumwidths = (right - left) * cumwidths + left
    cumwidths[..., 0] = left
    cumwidths[..., -1] = right
    widths = cumwidths[..., 1:] - cumwidths[..., :-1]

    derivatives = min_derivative + F.softplus(unnormalized_derivatives)

    heights = F.softmax(unnormalized_heights, dim=-1)
    heights = min_bin_height + (1 - min_bin_height * num_bins) * heights
    cumheights = torch.cumsum(heights, dim=-1)
    cumheights = F.pad(cumheights, pad=(1, 0), value=0.0)
    cumheights = (top - bottom) * cumheights + bottom
    cumheights[..., 0] = bottom
    cumheights[..., -1] = top
    heights = cumheights[..., 1:] - cumheights[..., :-1]

    if inverse:
        bin_idx = searchsorted(cumheights, inputs)[..., None]
    else:
        bin_idx = searchsorted(cumwidths, inputs)[..., None]

    input_cumwidths = cumwidths.gather(-1, bin_idx)[..., 0]
    input_bin_widths = widths.gather(-1, bin_idx)[..., 0]
    input_cumheights = cumheights.gather(-1, bin_idx)[..., 0]
    delta = heights / widths
    input_delta = delta.gather(-1, bin_idx)[..., 0]
    input_derivatives = derivatives.gather(-1, bin_idx)[..., 0]
    input_derivatives_plus_one = derivatives[..., 1:].gather(-1, bin_idx)[..., 0]
    input_heights = heights.gather(-1, bin_idx)[..., 0]

    if inverse:
        a = (inputs - input_cumheights) * (
            input_derivatives + input_derivatives_plus_one - 2 * input_delta
        ) + input_heights * (input_delta - input_derivatives)
        b = input_heights * input_derivatives - (inputs - input_cumheights) * (
            input_derivatives + input_derivatives_plus_one - 2 * input_delta
        )
        c = -input_delta * (inputs - input_cumheights)

        discriminant = b.pow(2) - 4 * a * c
        discriminant = torch.clamp(discriminant, min=0.0)
        root = (2 * c) / (-b - torch.sqrt(discriminant))
        outputs = root * input_bin_widths + input_cumwidths

        theta_one_minus_theta = root * (1 - root)
        denominator = input_delta + (
            (input_derivatives + input_derivatives_plus_one - 2 * input_delta) * theta_one_minus_theta
        )
        derivative_numerator = input_delta.pow(2) * (
            input_derivatives_plus_one * root.pow(2)
            + 2 * input_delta * theta_one_minus_theta
            + input_derivatives * (1 - root).pow(2)
        )
        logabsdet = -(torch.log(derivative_numerator) - 2 * torch.log(denominator))
        return outputs, logabsdet
    else:
        theta = (inputs - input_cumwidths) / input_bin_widths
        theta_one_minus_theta = theta * (1 - theta)

        numerator = input_heights * (input_delta * theta.pow(2) + input_derivatives * theta_one_minus_theta)
        denominator = input_delta + (
            (input_derivatives + input_derivatives_plus_one - 2 * input_delta) * theta_one_minus_theta
        )
        outputs = input_cumheights + numerator / denominator

        derivative_numerator = input_delta.pow(2) * (
            input_derivatives_plus_one * theta.pow(2)
            + 2 * input_delta * theta_one_minus_theta
            + input_derivatives * (1 - theta).pow(2)
        )
        logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)
        return outputs, logabsdet
