import numpy as np
import torch


class SDE(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def sigma(self, t: torch.Tensor):
        raise NotImplementedError

    def mean(self, t: torch.Tensor):
        raise NotImplementedError

    def perturb(self, x, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x)
        mean = self.mean(t)
        sigma = self.sigma(t)
        return mean * x + sigma * noise
    
    
class VpSdeCos(SDE):
    def __init__(self):
        self.t_min = 0.007
        self.t_max = 1 - 0.007

    def sigma_inverse(self, sigma: torch.tensor):
        return 1 / np.pi * torch.acos(1 - 2 * sigma)

    def sigma(self, t: torch.Tensor):
        return 0.5 * (1 - torch.cos(np.pi * t))

    def mean(self, t: torch.Tensor):
        return (1 - self.sigma(t)**2)**0.5

    def sigma_derivative(self, t: torch.Tensor):
        return 0.5 * np.pi * torch.sin(np.pi * t)

    def beta(self, t: torch.Tensor):
        return 2 * self.sigma(t) * self.sigma_derivative(t) / (
            1 - self.sigma(t)**2)

    def g(self, t: torch.Tensor):
        return self.beta(t)**0.5


class SubVpSdeCos(SDE):
    def __init__(self):
        self.t_min = 0.006
        self.t_max = 1 - 0.006

    def sigma_inverse(self, sigma: torch.tensor):
        return 1 / np.pi * torch.acos(1 - 2 * sigma)

    def sigma(self, t: torch.Tensor):
        return 0.5 * (1 - torch.cos(np.pi * t))

    def mean(self, t: torch.Tensor):
        return (1 - self.sigma(t))**0.5

    def sigma_from_mean_approx(self, mean_t, nu_t):
        return (1-mean_t**2-nu_t**2)

    def sigma_derivative(self, t: torch.Tensor):
        return 0.5 * np.pi * torch.sin(np.pi * t)

    def beta(self, t: torch.Tensor):
        return self.sigma_derivative(t) / (1 - self.sigma(t))

    def g(self, t: torch.Tensor):
        return (self.sigma_derivative(t) * self.sigma(t) *
                (2 - self.sigma(t)) / (1 - self.sigma(t)))**0.5