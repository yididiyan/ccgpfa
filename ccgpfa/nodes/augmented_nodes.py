"""
Module for augmented variables
"""
import torch
from torch.special import digamma
from scipy.special import digamma as sci_digamma

DIGAMMA_1 = sci_digamma(1)


class Omega:
    """
    Polya Gamma(PG) variables

    """

    def __init__(self, B, C=None):
        """

        :param B: Shape parameters of the PG variables
        :param C: Tilting parameters of the PG variables
        """
        self.B = B
        self.C = C or torch.zeros_like(self.B).to(self.B.device)  # initial values

    def update(self, B, C):
        assert self.B.shape == B.shape
        assert self.C.shape == C.shape

        self.B = B
        self.C = C
    def update_batch(self, B, C, update_indices):
        self.B[..., update_indices] = B
        self.C[..., update_indices] = C



    @property
    def mean(self):
        """
        Compute the mean of the PG variables

        :return: the mean
        """
        B = self.B
        C = self.C + 1e-7  # avoiding division by zero
        return 0.5 * (B / C) * torch.tanh(C / 2)

    def mean_(self, sample_indices):
        # slice with time index
        B = self.B[..., sample_indices]
        C = self.C[..., sample_indices] + 1e-7

        return 0.5 * (B / C) * torch.tanh(C / 2)



class Xi:
    """
    Polya Inverse Gamma(P-IG) variables
    """

    def __init__(self, C):
        """
        Initialize the P-IG variables with tilting parameters C
        :param C: tilting parameters of the variables
        """
        self.C = C

    @property
    def mean(self):
        C = self.C

        return 0.5 * (1 / C) * (digamma(C + 1) - DIGAMMA_1)

    def update(self, C):
        self.C = C


class Tau:
    """
    Gamma variables
    """

    def __init__(self, alpha):
        self.alpha = alpha
        self.beta = torch.ones_like(self.alpha).to(self.alpha.device)

    @property
    def mean(self):
        return self.alpha / self.beta

    def update(self, alpha):
        self.alpha = alpha


    def update_batch(self, alpha, update_indices):
        self.alpha[:, update_indices] = alpha

