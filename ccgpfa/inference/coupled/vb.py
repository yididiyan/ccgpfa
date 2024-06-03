import numpy as np
import torch
import tqdm

from scipy.stats import nbinom

from ccgpfa.nodes.latents import Latents
from ccgpfa.nodes.coupled_latents import CoupledLatents
from ccgpfa.nodes.weights import Weights
from ccgpfa.nodes.base_intensities import BaseIntensities
from ccgpfa.nodes.augmented_nodes import *
from ccgpfa.nodes.dispersions import Dispersions, DispersionsTruncated, DispersionsNumericalIntegration
from ccgpfa.utils import detach

"""
Not yet fully implemented 
"""
class VariationalBayesInference:

    def __init__(self, Y, D=10, lr=5e-2, device=None, ell_init=8., n_mc_samples=1, max_r=5., F_clamp=1.):
        """

        :param Y: data -- torch.Tensor -- number of trials x neurons x time steps
        :param D: # of latent dimensions
        :param lr: learning rate for the length scales
        :param device: device to run the computation
        :param ell_init: initial lengthscales
        :param n_mc_samples: number of MC samples to approximate mean of R
        :param max_r: maximum cap for R
        :param F_clamp: clamp magnitude for F
        """
        self.n_conditions = len(Y)

        ## Sum Y's acrros trials
        self.M = Y.shape[1] # number of trials
        self.Y = Y.sum(1)

        self.N, self.T = self.Y[0].shape

        self.F_clamp = F_clamp

        self.D = D
        # set up the nodes
        self.latents = CoupledLatents(self.n_conditions, self.D, self.T, device=device, ell=ell_init)
        self.latents.initialize()
        self.weights = Weights(self.N, self.D, device=device)
        self.base = BaseIntensities(self.N, device=device)
        self.dispersions = Dispersions(self.N, device=device, n_mc_samples=n_mc_samples, max_r=max_r)


        # augmented variables
        r_mean = self.dispersions.mean
        B = self.B
        self.tau = [Tau(B[c]) for c in range(self.n_conditions)]
        self.xi = Xi(r_mean) # same across trials
        self.omega = [Omega(B[c]) for c in range(self.n_conditions)]

        # jitter matrices
        self.jitter_weight = (1e-6 * torch.diag_embed(torch.ones(1, self.D).double())).to(device)
        self.jitter_latent = 1e-6 * torch.eye(self.T).double().to(device)

        # M optimizer
        self.optimizer = torch.optim.Adam([
            {
                'params': self.latents._ell, 'lr': lr
            }
        ])

        self.update_weight_prior = True



    @property
    def B(self):
        return [self.dispersions.mean[..., None] * self.M + self.Y[c] for c in range(self.n_conditions)]

    def Omega(self):
        """
        Sum of Omega mean vals across all trials
        :return:
        """
        return torch.stack([self.omega[c].mean for c in range(self.n_conditions) ])

    @property
    def F(self):
        return self.F_XW + self.F_base

    @property
    def F_XW(self):
        """
        Shape of (1 x N x D) @ (n_cond x D x T) = n_cond X N X T
        :return:
        """
        return self.weights.mean[None, ...] @ self.latents.mean

    @property
    def F_base(self):
        return self.base.mean[None, ...]

    @property
    def success_prob(self):
        return 1 / (1 + torch.exp(-self.F))

    @property
    def Kappa(self):
        """
        n_cond x N X T shape tensor
        :return:
        """
        return torch.stack([0.5 * (self.Y[c] - self.M * self.dispersions.mean[..., None]) for c in range(self.n_conditions)])

    @property
    def E_X_squared(self):
        """
        E[X] E[X]^T  -- D X D shape tensor
        """
        return self.latents.mean @ self.latents.mean.transpose(-1, -2)

    def E_F_squared(self):
        E_F_base_squared = self.base.quadratic_expectation[None, ...]
        E_F_cross = self.F_base * self.F_XW

        # E_F_XW_squared

        # n_cond x D x D x T
        E_X_X = self.latents.quadratic_expectation

        # Quadratic form
        # E[F] = E[W E[XX] W] = Tr(E[XX] Cov(W)) + E[W] E[XX] E[W]
        # Quadatic term
        quad_term = torch.einsum('nd,bdet->bnet', self.weights.mean, E_X_X)
        # E[W] E[XX] E[W]
        quad_term = (quad_term * self.weights.mean.unsqueeze(-1).unsqueeze(0)).sum(2)  # sum across latent dimensions => N X T
        trace_term = torch.diagonal(torch.einsum('bdet,nef->bndft', E_X_X, self.weights.covariance)
                                    , dim1=2, dim2=3).sum(-1)
        E_WXXW = quad_term + trace_term

        return E_F_base_squared + E_WXXW + 2 * E_F_cross

    def update_base(self):
        """
        Update base intensities
        :return:
        """
        Omega = self.Omega()
        precision = 1 / self.base.prior_covariance + Omega.sum(-1, keepdims=True).sum(0)
        updated_covariance = 1 / precision
        updated_mean = updated_covariance * (self.Kappa - self.F_XW * Omega).sum(-1, keepdims=True).sum(0) # sum across conditions

        self.base.update(updated_mean, updated_covariance)
        self.update_base_precision()

    def update_base_precision(self):
        # update the prior covariance
        alpha = 1e-3 + self.N / 2
        beta = 1e-3 + 0.2 * self.base.quadratic_expectation.sum(0, keepdims=True)
        self.base.prior_covariance = beta / alpha

    def update_weights(self):
        Omega = self.Omega()
        Base = self.F_base

        # D x D latent mean product E[X] E[X]^T
        E_X_X_T = self.latents.quadratic_expectation

        covariance_expectation_term = torch.einsum(
            'bdet,bnt->nde',
            E_X_X_T,
            Omega
        )

        updated_covariance = torch.linalg.inv(
            self.weights.prior_precision + covariance_expectation_term + self.jitter_weight)

        # N x D
        mean_term = ((self.Kappa - Base * Omega) @ self.latents.mean.transpose(-1, -2) ).sum(0)
        updated_mean = (updated_covariance @ mean_term.unsqueeze(-1)).squeeze(-1)

        self.weights.update(updated_mean, updated_covariance)
        self.update_weight_precision()

    def update_weight_precision(self):
        if self.update_weight_prior:
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

        r_mean = self.dispersions.mean
        r_var = self.dispersions.variance

        E_r_squared = torch.square(r_mean) + r_var

        # update omega
        for c in range(self.n_conditions):
            self.omega[c].update( B[c], torch.sqrt(E_F_squared[c]))
            self.tau[c].update(B[c])

        self.xi.update(torch.sqrt(E_r_squared))

    def update_dispersion(self):
        Xi = self.xi.mean

        updated_p = self.n_conditions * self.M * self.T * torch.ones_like(self.dispersions.p)
        updated_a = self.n_conditions * self.M * self.T * Xi
        updated_b = self.n_conditions * self.M * self.T * (np.euler_gamma - np.log(2.))
        E_log_tau = torch.stack([torch.digamma(self.tau[c].alpha) for c in range(self.n_conditions)]).sum(0)

        F = self.F.sum(0) # sum across conditions

        F = torch.clamp(F, min=-self.F_clamp, max=self.F_clamp)
        F = F * self.M # multiply by the number of trials


        updated_b += (-0.5 * F + E_log_tau).sum(-1)

        self.dispersions.update(updated_p, updated_a, updated_b)

    def update_latents(self):

        E_W_W = self.weights.quadratic_expectation_diag()
        Omega = self.Omega()

        Kappa = self.Kappa
        prior_K_inv = self.latents.prior_K_inv().detach()

        for d in range(self.D):
            tmp_d = (E_W_W[:, d][None, ..., None] * Omega).sum(1)

            updated_cov = torch.linalg.inv(prior_K_inv[:, d, ...].detach() + torch.diag_embed(tmp_d) + self.jitter_latent[None, ...])

            # TODO: add assertions
            assert torch.all(torch.diagonal(updated_cov, dim1=-2, dim2=-1) > 0.).item()

            total_effect = self.F
            other_effects = total_effect - self.weights.mean[None, :, d:d + 1] @ self.latents.mean[:, d:d + 1, :]

            mean_term_d = ((Kappa - other_effects * Omega).transpose(-1, -2) @ self.weights.mean[None, :, d:d + 1])

            updated_mean = (updated_cov @ mean_term_d).squeeze(-1)

            # zero-center latents
            updated_mean = updated_mean - updated_mean.mean(1, keepdims=True)
            self.latents.update(updated_mean, updated_cov, d)

        loss = - self.latents()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.latents.prior_K(update=True)
        self.latents.prior_K_inv(update=True)



def generate(n_cond, D=1, T=100, N=20,  seed=1, shift=0., n_samples=5, max_r=5.):
    np.random.seed(seed)

    # temporally correlated features
    t = np.arange(0, T)
    K = [np.exp(-(1. / (2 * l ** 2)) * np.square(t[..., None] - t[None, ...])) for l in [20, 20, 20, 20, 20]]
    X = [np.array([np.random.multivariate_normal(np.zeros_like(t), K[i]) for i in range(D)]).T for _ in range(n_cond)]
    X = [X[0] for _ in range(n_cond)]

    # Zero out the first latent for condition 1
    X[0][:, 0] = X[0][:, 0] * -0.25


    # Generate coefficients
    weights = np.random.randn(D, N)

    F = [X[i] @ weights - shift for i in range(n_cond)]
    p_success = [1 / (1 + np.exp(-F[i])) for i in range(n_cond)]  # success probability

    r_true = np.array([np.random.uniform(0.5, max_r) for i in range(N)])

    Y = []
    for j in range(n_cond):
        Y_cond = []
        for i in range(n_samples):
            Y_cond.append(nbinom.rvs(r_true[None, ...], p_success[j]).T)
        Y.append(np.array(Y_cond))

    return Y, F, K, X, weights, r_true

