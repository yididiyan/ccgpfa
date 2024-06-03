import numpy as np
import torch
import random

from scipy.stats import nbinom

from ccgpfa.nodes.latents import Latents, SparseLatents, SparseStochasticLatent
from ccgpfa.nodes.weights import Weights
from ccgpfa.nodes.base_intensities import BaseIntensities
from ccgpfa.nodes.augmented_nodes import *
from ccgpfa.nodes.dispersions import Dispersions
from ccgpfa.utils import detach
from ccgpfa.inference.vb_neg_binomial_multi_trial import VariationalBayesInference, SparseVariationalInference, generate

class StochasticSparseVariationalInference(VariationalBayesInference):


    def __init__(self, Y, D=10, M=5, lr=5e-2, device='cuda', ell_init=8., ngd_lr=0.6, batch_size=10, **kwargs):
        T = Y.shape[-1]  # M trials
        init_sample_indices = list(range(0, batch_size))
        self.latents = SparseStochasticLatent(D, T, M=M, device=device, ell=ell_init, sample_indices=init_sample_indices)
        self.latents.initialize()
        self.ngd_lr = ngd_lr
        # self.base_lr = 1.

        super(StochasticSparseVariationalInference, self).__init__(Y, D=D, lr=lr, device=device, ell_init=ell_init, latents=self.latents, **kwargs)


        print(f'Using {self.latents.M} inducing points -- nat. learning rate {ngd_lr}')
        self.jitter_inducing = (1e-6 * torch.diag_embed(torch.ones(self.latents.M).double())).to(device)


        self.batch_size=batch_size
        # sample indices
        self.all_indices = list(range(0, self.T))
        random.seed(0)
        random.shuffle(self.all_indices)
        self.sample_indices = init_sample_indices
        # self.sample_indices = self.update_sample_indices()

        self.time = 1

    @property
    def gradient_scale(self):
        return self.T / self.batch_size

    def update_sample_indices(self, shuffle=True):
        self.time += 1
        if len(self.all_indices) < self.batch_size:
            # reinit random list
            # all_indices = list(range(0, self.T))

            # if shuffle:
            #     all_indices = np.array(all_indices).reshape(self.M, -1)
            #     all_indices = np.random.default_rng().permuted(all_indices, axis=1).T.reshape(-1)

            # self.all_indices = all_indices
            #
            self.all_indices = list(range(0, self.T))
            if shuffle: random.shuffle(self.all_indices)

        indices = self.all_indices[:self.batch_size]
        self.all_indices = self.all_indices[self.batch_size:]

        # Update the latent nodes - this initializes all necessary prior matrices
        self.latents.update_sample_indices(indices)
        self.sample_indices = indices
        # self.ngd_lr = self.base_lr / (1 + 0.05 * self.time) # decay rate
        # self.ngd_lr = 1 - 0.01  ** (1 / self.time)
        # print(self.ngd_lr)



    def B_(self):
        if self.tied_trials:
            return self.M * self.dispersions.mean[..., None] + self.Y_sum
        return [self.dispersions.mean[..., None] + self.Y[m][..., self.sample_indices] for m in range(self.M)]

    def Omega(self):
        """
        Sum of Omega mean vals across all trials
        :return:
        """
        if self.tied_trials:
            return self.omega.mean_(sample_indices=self.sample_indices)
        return torch.stack([self.omega[m].mean_(sample_indices=self.sample_indices) for m in range(self.M) ]).sum(0)


    def Kappa(self):
        return 0.5 * (self.Y_sum[..., self.sample_indices] - self.M * self.dispersions.mean[..., None])
    def update_base(self):
        """
        Update base intensities
        :return:
        """
        Omega = self.Omega()
        eta_2 = -0.5 * ( 1 / self.base.prior_covariance + self.gradient_scale * Omega.sum(1, keepdims=True))
        eta_1 = self.gradient_scale * (self.Kappa() - self.F_XW * Omega).sum(1, keepdims=True)

        self.base.update_natural(eta_1, eta_2, step_size=self.ngd_lr)


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

        eta_2 = -0.5 * (self.weights.prior_precision + self.gradient_scale * covariance_expectation_term)

        eta_1 = self.gradient_scale * (self.Kappa() - Base * Omega) @ self.latents.mean.T


        self.weights.update_natural(eta_1, eta_2, step_size=self.ngd_lr)

        # update prior precision -- ARD Gamma variables
        alpha = 1e-5 + 0.5 * self.N
        beta = 1e-5 + 0.5 * self.weights.quadratic_expectation_diag().sum(0)
        self.weights.update_prior_precision(torch.diag_embed(alpha / beta).unsqueeze(0))


    def update_augmented_vars(self):
        """
        Update augmented variables -- Omega & tau
        """

        B = self.B_()
        E_F_squared = self.E_F_squared()



        # update omega
        if self.tied_trials:
            assert B.shape == E_F_squared.shape
            self.omega.update_batch(B,  torch.sqrt(E_F_squared), self.sample_indices)
            self.tau.update_batch(B, self.sample_indices)
        else:
            # update for each trial independently
            for m in range(self.M):
                assert B[m].shape == E_F_squared.shape
                self.omega[m].update_batch( B[m], torch.sqrt(E_F_squared), self.sample_indices)
                self.tau[m].update_batch( B[m], self.sample_indices)

    def update_dispersion(self):
        indices = list(range(0, self.T))
        self.sample_indices = indices
        self.latents.update_sample_indices(indices)


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

        # update xi
        r_mean = self.dispersions.mean
        r_var = self.dispersions.variance
        E_r_squared = torch.square(r_mean) + r_var
        self.xi.update(torch.sqrt(E_r_squared))

        self.update_sample_indices()

    def update_latents(self):
        E_W_W = self.weights.quadratic_expectation_diag()
        Omega = self.Omega()

        Kappa = self.Kappa()
        prior_K_mm_inv = self.latents.prior_K_mm_inv().detach()
        prior_K_mm_inv_K_mt = self.latents.prior_K_mm_inv_K_mt().detach()

        for d in range(self.D):
            tmp_d = (E_W_W[:, d][..., None] * Omega).sum(0)

            # updated_cov = torch.linalg.inv(prior_K_inv[d].detach() + torch.diag(tmp_d) + self.jitter_latent)

            K_mm_inv = prior_K_mm_inv[d, :, :]
            K_mm_inv_K_mt = prior_K_mm_inv_K_mt[d, :, :]

            # TODO: add assertions

            # S = Kmm^{-1} + Kmm^{-1} K_mt \sum_n (E[w_n,d] E[Omega_n]) (Kmm^{-1} K_mt)^T
            eta_2 = -0.5 * (K_mm_inv
                            + self.gradient_scale * (K_mm_inv_K_mt
                                  @ torch.diag_embed(tmp_d)
                                  @  K_mm_inv_K_mt.transpose(-1, -2)))


            total_effect = self.F
            other_effects = total_effect - self.weights.mean[:, d:d + 1] @ self.latents.mean[d:d + 1, :]

            mean_term_d = self.gradient_scale * ((Kappa - other_effects * Omega).T @ self.weights.mean[:, d:d + 1]).squeeze(-1)

            eta_1 = K_mm_inv_K_mt @  mean_term_d

            self.latents.update_natural(eta_1, eta_2, d, step_size=self.ngd_lr)

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

        self.latents.prior_K_mm(update=True)
        self.latents.prior_K_mm_inv(update=True)



