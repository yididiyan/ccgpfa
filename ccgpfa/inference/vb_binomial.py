import numpy as np
import torch

from scipy.stats import binom

from ccgpfa.nodes.latents import Latents
from ccgpfa.nodes.weights import Weights
from ccgpfa.nodes.base_intensities import BaseIntensities
from ccgpfa.nodes.augmented_nodes import *
from ccgpfa.nodes.dispersions import Dispersions
from ccgpfa.utils import detach


class VariationalBayesInference:

    def __init__(self, Y, D=10, lr=5e-2, device=None, n_mc_samples=1):
        self.Y = Y.to(device)
        self.N, self.T = self.Y.shape

        self.D = D
        # set up the nodes
        self.latents = Latents(self.D, self.T, device=device)
        self.latents.initialize()
        self.weights = Weights(self.N, self.D, device=device)
        self.base = BaseIntensities(self.N, device=device)

        # the maximum number of spikes across neurons
        self.n = self.Y.max(-1)[0].double()

        B = self.B
        self.omega = Omega(B)

        # jitter matrices
        self.jitter_weight = (1e-6 * torch.diag_embed(torch.ones(1, self.D).double())).to(device)

        # M optimizer
        self.optimizer = torch.optim.Adam([
            {
                'params': self.latents._ell, 'lr': lr
            }
        ])

    @property
    def B(self):
        return self.n[..., None].expand(self.N, self.T)

    @property
    def F(self):
        return self.F_XW + self.F_base

    @property
    def F_XW(self):
        """
        Shape of (N x D) X (D x T) = N X T
        :return:
        """
        return self.weights.mean @ self.latents.mean

    @property
    def F_base(self):
        return self.base.mean

    @property
    def success_prob(self):
        return 1 / (1 + torch.exp(-self.F))

    @property
    def Kappa(self):
        """
        N X T shape tensor
        :return:
        """
        # return 0.5 * (self.Y - self.n[..., None])
        return (self.Y - 0.5 * self.n[..., None])

    @property
    def E_X_squared(self):
        """
        E[X] E[X]^T  -- D X D shape tensor
        """
        return self.latents.mean @ self.latents.mean.T

    def E_F_squared(self):
        E_F_base_squared = self.base.quadratic_expectation
        E_F_cross = self.base.mean * self.F_XW

        # E_F_XW_squared

        # D x D x T
        E_X_X = self.latents.quadratic_expectation

        # Quadratic form
        # E[F] = E[W E[XX] W] = Tr(E[XX] Cov(W)) + E[W] E[XX] E[W]
        # Quadatic term
        quad_term = torch.einsum('nd,det->net', self.weights.mean, E_X_X)
        # E[W] E[XX] E[W]
        quad_term = (quad_term * self.weights.mean.unsqueeze(-1)).sum(1)  # sum across latent dimensions => N X T
        trace_term = torch.diagonal(torch.einsum('det,nef->ndft', E_X_X, self.weights.covariance),
                                    dim1=1, dim2=2).sum(-1)
        E_WXXW = quad_term + trace_term

        return E_F_base_squared + E_WXXW + 2 * E_F_cross

    def update_base(self):
        """
        Update base intensities
        :return:
        """
        Omega = self.omega.mean
        precision = 1 / self.base.prior_covariance + Omega.sum(1, keepdims=True)
        updated_covariance = 1 / precision
        updated_mean = updated_covariance * (self.Kappa - self.F_XW * Omega).sum(1, keepdims=True)

        self.base.update(updated_mean, updated_covariance)
        # update the prior covariance
        alpha = 1e-3 + self.N / 2
        beta = 1e-3 + 0.2 * self.base.quadratic_expectation.sum(0, keepdims=True)
        self.base.prior_covariance = beta / alpha

    def update_weights(self):
        Omega = self.omega.mean
        Base = self.base.mean

        # D x D latent mean product E[X] E[X]^T
        E_X_X_T = self.latents.quadratic_expectation

        covariance_expectation_term = torch.einsum(
            'det,nt->nde',
            E_X_X_T,
            Omega
        )

        updated_covariance = torch.linalg.inv(
            self.weights.prior_precision + covariance_expectation_term + self.jitter_weight)

        # N x D
        mean_term = (self.Kappa - Base * Omega) @ self.latents.mean.T.double()
        updated_mean = (updated_covariance @ mean_term.unsqueeze(-1)).squeeze(-1)

        self.weights.update(updated_mean, updated_covariance)
        # update prior precision -- ARD Gamma variables
        alpha = 1e-3 + 0.5 * self.N
        beta = 1e-3 + 0.5 * self.weights.quadratic_expectation_diag().sum(0)
        self.weights.update_prior_precision(torch.diag_embed(alpha / beta).unsqueeze(0))



    def update_augmented_vars(self):
        """
        Update augmented variables -- Omega, Xi, Tau
        """
        B = self.B
        E_F_squared = self.E_F_squared()

        # update omega
        self.omega.update(B, torch.sqrt(E_F_squared))


    def update_latents(self):

        E_W_W = self.weights.quadratic_expectation_diag()
        Omega = self.omega.mean

        Kappa = self.Kappa
        jitter = 1e-6 * torch.eye(self.T).double()
        prior_K_inv = self.latents.prior_K_inv().detach()

        for d in range(self.D):
            tmp_d = (E_W_W[:, d][..., None] * Omega).sum(0)

            updated_cov = torch.linalg.inv(prior_K_inv[d].detach() + torch.diag(tmp_d) + jitter)

            # TODO: add assertions

            total_effect = self.F
            other_effects = total_effect - self.weights.mean[:, d:d + 1] @ self.latents.mean[d:d + 1, :]

            mean_term_d = ((Kappa - other_effects * Omega).T @ self.weights.mean[:, d:d + 1]).squeeze(-1)

            updated_mean = updated_cov @ mean_term_d

            # zero-center latents
            updated_mean = updated_mean - updated_mean.mean()
            self.latents.update(updated_mean, updated_cov, d)

        loss = - self.latents()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.latents.prior_K(update=True)
        self.latents.prior_K_inv(update=True)



def generate(D=1, T=100, N=20,  seed=1, n=2, shift=0.):
    np.random.seed(seed)

    # Generate features
    # random features
    # X = np.random.randn(T, D)

    # temporally correlated features
    t = np.arange(0, T)
    K = [np.exp(-(1. / (2 * l ** 2)) * np.square(t[..., None] - t[None, ...])) for l in [20, 20, 20, 20, 20]]
    X = np.array([np.random.multivariate_normal(np.zeros_like(t), K[i]) for i in range(D)]).T

    # Generate coefficients
    weights = np.random.randn(D, N)

    F = X @ weights - shift
    p_success = 1 / (1 + np.exp(-F))  # success probability

    Y = binom.rvs(n, p_success)

    return Y, F, K, X, weights, n




