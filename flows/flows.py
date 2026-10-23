import torch
import torch.distributions as dist
import torch.nn as nn
import zuko
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent / 'src'))

from .utils import VarianceGamma

class GaussianNormFlow(nn.Module):
    def __init__(self, dim=1, num_layers=4, num_bins=8, tail_bound=5.0):
        super().__init__()

        # Creta standard normal distribution
        self.base = dist.Normal(torch.zeros(dim), torch.ones(dim))

        # Build the Neural spline flows layers
        self.flow = zuko.flows.NSF(
            features=dim,
            transforms=num_layers,
            bins=num_bins,
            tail_bound=tail_bound,
            hidden_features=(64, 64)
        )

        self.flow_instance = self.flow() 

    def log_prob(self, x):
        return self.flow_instance.log_prob(x)

    def sample(self, n):
        return self.flow_instance.sample((n,))
    