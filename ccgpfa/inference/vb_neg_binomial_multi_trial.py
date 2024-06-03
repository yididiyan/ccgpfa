import numpy as np
import torch

from scipy.stats import nbinom
from sklearn import decomposition

from ccgpfa.nodes.latents import Latents, SparseLatents
from ccgpfa.nodes.weights import Weights
from ccgpfa.nodes.base_intensities import BaseIntensities
from ccgpfa.nodes.augmented_nodes import *
from ccgpfa.nodes.dispersions import Dispersions, DispersionsTruncated, DispersionsNumericalIntegration
from ccgpfa.utils import detach


class VariationalBayesInference:

    def __init__(self, Y, D=10, lr=5e-2, device=None, normalize=None, ell_init=8., n_mc_samples=10, max_r=5., F_clamp=1., latents=None, tied_trials=False, **kwargs):
        """

        :param Y: data -- torch.Tensor -- number of trials x neurons x time steps
        :param D: # of latent dimensions
        :param lr: learning rate for the length scales
        :param device: device to run the computation
        :param normalize: normalize flag
        :param ell_init: initial lengthscales
        :param n_mc_samples: number of MC samples to approximate mean of R
        :param max_r: maximum cap for R
        :param F_clamp: clamp magnitude for F
        """
        self.Y = Y.to(device)
        self.M, self.N, self.T = self.Y.shape # M trials
        self.normalize = normalize
        self.F_clamp = F_clamp

        self.D = D

        # initialize weights using FA
        mod = decomposition.FactorAnalysis(n_components=self.D)
        Y_fa = detach(self.Y).transpose(0, 2, 1).reshape(self.M * self.T, self.N)
        mod.fit_transform(Y_fa)
        weight_init = torch.tensor(mod.components_.T)  # (n x d)
        # weight_init = None

        # set up the nodes
        self.latents = latents or Latents(self.D, self.T, device=device, ell=ell_init)
        self.latents.initialize()
        self.weights = Weights(self.N, self.D, init_weight=weight_init, device=device)
        self.base = BaseIntensities(self.N, device=device)
        self.dispersions = Dispersions(self.N, device=device, n_mc_samples=n_mc_samples, max_r=max_r)
        self.dispersions._mean = self.Y.amax(dim=(0, -1)).double() + 1
        print(self.dispersions._mean)

        self.Y_sum = self.Y.sum(0)
        self.tied_trials = tied_trials
        self.zero_centered = kwargs.get('zero_centered', True)
        print('Zero centered ', self.zero_centered)

        # augmented variables
        r_mean = self.dispersions.mean
        B = self.B

        self.xi = Xi(r_mean) # same across trials

        if self.tied_trials:
            self.omega = Omega(B)
            self.tau = Tau(B)
        else:
            self.omega = [Omega(B[m]) for m in range(self.M)]
            self.tau = [Tau(B[m]) for m in range(self.M)]

        # update initial values of the a, p, b
        # import ipdb;
        # ipdb.set_trace()
        # updated_p = self.M * self.T * torch.ones_like(self.dispersions.p)
        # updated_a = self.M * self.T * self.xi.mean  # N component
        # updated_b = self.M * self.T * (np.euler_gamma - np.log(2.)) * torch.ones_like(self.dispersions.b)
        # if self.tied_trials:
        #     E_log_tau = self.tau.alpha[..., self.sample_indices]
        # else:
        #     E_log_tau = torch.stack(
        #         [torch.digamma(self.tau[m].alpha) for m in range(self.M)]).sum(0)
        # updated_b += (E_log_tau).sum(-1)  # note without the F
        # self.dispersions.update(updated_p, updated_a, updated_b)
        # / updated


        # jitter matrices
        self.jitter_weight = (1e-6 * torch.diag_embed(torch.ones(1, self.D).double())).to(device)
        self.jitter_latent = 1e-6 * torch.eye(self.T).double().to(device)

        # M optimizer
        if self.latents.__class__.__name__ in [ 'SparseLatents', 'SparseStochasticLatent']:
            self.optimizer = torch.optim.Adam([
                {
                    'params': self.latents._ell, 'lr': lr
                }
                ,
                {
                    'params': self.latents.ts_mm, 'lr': lr
                }
            ])
        else:
            self.optimizer = torch.optim.Adam([
                {
                    'params': self.latents._ell, 'lr': lr
                }
            ])
        self.update_weight_prior = True



    @property
    def B(self):
        if self.tied_trials:
            return self.M * self.dispersions.mean[..., None] + self.Y_sum
        return [self.dispersions.mean[..., None] + self.Y[m] for m in range(self.M)]

    def Omega(self):
        """
        Sum of Omega mean vals across all trials
        :return:
        """
        if self.tied_trials:
            return self.omega.mean

        return torch.stack([self.omega[m].mean for m in range(self.M) ]).sum(0)

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
    def firing_rate(self):
        return torch.exp(self.F) * self.dispersions.mean[..., None]

    def loglikelihood(self, data):
        """
        data
        :param data: Batch x N x T
        :return:
        """
        prob = detach(self.success_prob)[None, ...]
        return nbinom.logpmf(data, detach(self.dispersions.mean)[None, ..., None], 1 - prob)

    @property
    def Kappa(self):
        """
        N X T shape tensor
        :return:
        """
        return 0.5 * (self.Y_sum - self.M * self.dispersions.mean[..., None])

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

        tmp = E_F_base_squared + E_WXXW + 2 * E_F_cross
        tmp[tmp < 0.] = 1e-8

        return tmp

    def update_base(self):
        """
        Update base intensities
        :return:
        """
        Omega = self.Omega()
        precision = 1 / self.base.prior_covariance + Omega.sum(1, keepdims=True)
        updated_covariance = 1 / precision
        updated_mean = updated_covariance * (self.Kappa - self.F_XW * Omega).sum(1, keepdims=True)

        self.base.update(updated_mean, updated_covariance)
        self.update_base_precision()

    def update_base_precision(self):
        # update the prior covariance
        alpha = 1e-5 + self.N / 2
        beta = 1e-5 + 0.5 * self.base.quadratic_expectation.sum(0, keepdims=True)
        self.base.prior_covariance = beta / alpha

    def update_weights(self):
        Omega = self.Omega()
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
        self.update_weight_precision()

    def update_weight_precision(self):
        if self.update_weight_prior:
            # update prior precision -- ARD Gamma variables
            alpha = 1e-5 + 0.5 * self.N
            beta = 1e-5 + 0.5 * self.weights.quadratic_expectation_diag().sum(0)
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
        if self.tied_trials:
            self.omega.update(B, torch.sqrt(E_F_squared))
            self.tau.update(B)
        else:
            for m in range(self.M):
                self.omega[m].update( B[m], torch.sqrt(E_F_squared))
                self.tau[m].update(B[m])

        self.xi.update(torch.sqrt(E_r_squared))

    def update_dispersion(self):
        Xi = self.xi.mean

        updated_p = self.M * self.T * torch.ones_like(self.dispersions.p)
        updated_a = self.M * self.T * Xi
        updated_b = self.M * self.T * (np.euler_gamma - np.log(2.))
        if self.tied_trials:
            E_log_tau = self.tau.alpha
        else:
            E_log_tau = torch.stack([torch.digamma(self.tau[m].alpha) for m in range(self.M)]).sum(0)

        F = self.F

        F = torch.clamp(F, min=-self.F_clamp, max=self.F_clamp)
        F = F * self.M


        updated_b += (-0.5 * F + E_log_tau).sum(-1)

        self.dispersions.update(updated_p, updated_a, updated_b)

    def update_latents(self):

        E_W_W = self.weights.quadratic_expectation_diag()
        Omega = self.Omega()

        Kappa = self.Kappa
        prior_K_inv = self.latents.prior_K_inv().detach()

        for d in range(self.D):
            tmp_d = (E_W_W[:, d][..., None] * Omega).sum(0)

            updated_cov = torch.linalg.inv(prior_K_inv[d].detach() + torch.diag(tmp_d) + self.jitter_latent)

            assert torch.all(torch.diag(
                updated_cov) > 0.).item(), 'WARN:  diagonal values of the covariance are not all non-negative'

            total_effect = self.F
            other_effects = total_effect - self.weights.mean[:, d:d + 1] @ self.latents.mean[d:d + 1, :]

            mean_term_d = ((Kappa - other_effects * Omega).T @ self.weights.mean[:, d:d + 1]).squeeze(-1)

            updated_mean = updated_cov @ mean_term_d

            # zero-center latents
            if self.zero_centered:
                updated_mean = updated_mean - updated_mean.mean()
            self.latents.update(updated_mean, updated_cov, d)


        loss = - self.latents()
        old_ell = self.latents._ell.detach().clone()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if not torch.all(self.latents.ell > 0.):
            new_ell = self.latents._ell.detach()
            new_ell[torch.isnan(new_ell)] = old_ell[torch.isnan(new_ell)]
            with torch.no_grad():
                self.latents._ell.copy_(new_ell)

        self.latents.prior_K(update=True)
        self.latents.prior_K_inv(update=True)




class SparseVariationalInference(VariationalBayesInference):


    def __init__(self, Y, D=10, M=5, lr=5e-2, device=None, ell_init=8., n_mc_samples=1, max_r=5., F_clamp=1., **kwargs):
        T = Y.shape[-1]
        self.latents = SparseLatents(D, T, M=M, device=device, ell=ell_init)
        super(SparseVariationalInference, self).__init__(Y, D=D, lr=lr, device=device, ell_init=ell_init, n_mc_samples=n_mc_samples, max_r=max_r, F_clamp=F_clamp, latents=self.latents )
        # self.M = M

        self.jitter_inducing = (1e-6 * torch.diag_embed(torch.ones(self.latents.M).double())).to(device)

    def update_latents(self):
        E_W_W = self.weights.quadratic_expectation_diag()
        Omega = self.Omega()

        Kappa = self.Kappa
        prior_K_mm_inv = self.latents.prior_K_mm_inv().detach()

        for d in range(self.D):
            tmp_d = (E_W_W[:, d][..., None] * Omega).sum(0)

            # updated_cov = torch.linalg.inv(prior_K_inv[d].detach() + torch.diag(tmp_d) + self.jitter_latent)

            K_mm_inv = self.latents.prior_K_mm_inv()[d, :, :].detach()
            K_mm_inv_K_mt = self.latents.prior_K_mm_inv_K_mt()[d, :, :].detach()


            # S = Kmm^{-1} + Kmm^{-1} K_mt \sum_n (E[w_n,d] E[Omega_n]) (Kmm^{-1} K_mt)^T
            updated_cov = K_mm_inv + K_mm_inv_K_mt @ torch.diag_embed(tmp_d) @  K_mm_inv_K_mt.transpose(-1, -2)
            updated_cov = torch.linalg.inv(updated_cov + self.jitter_inducing)  # take an inverse

            assert torch.all(torch.diag(
                updated_cov) > 0.).item(), 'WARN:  diagonal values of the covariance are not all non-negative'

            total_effect = self.F
            other_effects = total_effect - self.weights.mean[:, d:d + 1] @ self.latents.mean[d:d + 1, :]

            mean_term_d = ((Kappa - other_effects * Omega).T @ self.weights.mean[:, d:d + 1]).squeeze(-1)

            updated_mean = updated_cov @ K_mm_inv_K_mt @  mean_term_d

            # zero-center latents
            if self.zero_centered:
                updated_mean = updated_mean - updated_mean.mean()
            self.latents.update(updated_mean, updated_cov, d)

        loss = - self.latents()
        old_ell = self.latents._ell.detach().clone()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if not torch.all(self.latents.ell > 0.):
            new_ell = self.latents._ell.detach()
            new_ell[torch.isnan(new_ell)] =  old_ell[torch.isnan(new_ell)]
            with torch.no_grad():
                self.latents._ell.copy_(new_ell)

        self.latents.prior_K_tt(update=True)
        self.latents.prior_K_mm(update=True)
        self.latents.prior_K_mt(update=True)
        self.latents.prior_K_mm_inv(update=True)
        self.latents.prior_K_mm_inv_K_mt(update=True)





def generate(D=1, T=100, N=20,  seed=1, shift=0.5, n_samples=5, max_r=5.):
    np.random.seed(seed)

    # temporally correlated features
    t = np.arange(0, T)
    K = [np.exp(-(1. / (2 * l ** 2)) * np.square(t[..., None] - t[None, ...])) for l in [20, 20, 20, 20, 20]]
    X = np.array([np.random.multivariate_normal(np.zeros_like(t), K[i]) for i in range(D)]).T

    # Generate coefficients
    weights = np.random.randn(D, N) * 0.5

    F = X @ weights + shift
    p_success = 1 / (1 + np.exp(-F))  # success probability

    r_true = np.array([np.random.uniform(0.5, max_r) for i in range(N)])

    Y = []
    for i in range(n_samples):
        Y.append(nbinom.rvs(r_true[None, ...], p_success).T)

    return Y, F.T, K, X, weights, r_true

