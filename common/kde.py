import torch
import torch.nn as nn
import numpy as np
import abc

class Kernel(abc.ABC, nn.Module):
    """Base class which defines the interface for all kernels."""

    def __init__(self, bandwidth=1.0):
        """Initializes a new Kernel.

        Args:
            bandwidth: The kernel's (band)width.
        """
        super().__init__()
        self.bandwidth = bandwidth

    def _diffs(self, test_Xs, train_Xs):
        """Computes difference between each x in test_Xs with all train_Xs."""
        # print('Test Xs shape: ', test_Xs.shape)
        test_Xs = test_Xs.reshape(test_Xs.shape[0], 1, *test_Xs.shape[1:])
        train_Xs = train_Xs.reshape(1, train_Xs.shape[0], *train_Xs.shape[1:])
        # test_Xs = test_Xs.view(test_Xs.shape[0], 1, *test_Xs.shape[1:])
        # train_Xs = train_Xs.view(1, train_Xs.shape[0], *train_Xs.shape[1:])
        return test_Xs - train_Xs

    @abc.abstractmethod
    def forward(self, test_Xs, train_Xs):
        """Computes log p(x) for each x in test_Xs given train_Xs."""

    @abc.abstractmethod
    def sample(self, train_Xs):
        """Generates samples from the kernel distribution."""


class GaussianKernel(Kernel):
    """Implementation of the Gaussian kernel."""

    def forward(self, test_Xs):
        n, d = self.train_Xs.shape
        n, h = torch.tensor(n, dtype=torch.float32), torch.tensor(self.bandwidth)
        pi = torch.tensor(np.pi)

        Z = 0.5 * d * torch.log(2 * pi) + d * torch.log(h) + torch.log(n)
        weights = self.weights.reshape(1, -1, 1)
        diffs = weights * self._diffs(test_Xs, self.train_Xs) / h
        log_exp = -0.5 * torch.norm(diffs, p=2, dim=-1) ** 2

        return torch.logsumexp(log_exp - Z, dim=-1)

    def fit(self, train_Xs, weights=None):
        self.train_Xs = train_Xs
        if weights is not None:
            self.weights = weights
        else:
            self.weights = torch.ones(train_Xs.shape[0])

    @torch.no_grad()
    def sample(self, train_Xs):
        device = train_Xs.device
        noise = torch.randn(train_Xs.shape, device=device) * self.bandwidth
        return train_Xs + noise