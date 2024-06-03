import torch
import numpy as np
from scipy.stats import nbinom
from ccgpfa.inference.vb_neg_binomial_multi_trial import VariationalBayesInference
from ccgpfa.nodes.latents import Latents
from ccgpfa.nodes.weights import Weights
from ccgpfa.nodes.augmented_nodes import *
from ccgpfa.utils import detach
from experiments.utils import load_data, split_data

"""
Not yet fully implemented 
"""
class JointDatasetVariationalBayesInference:

    def __init__(self, Y, D_exclusive, D_shared, **kwargs):
        """

        :param Y: list of tensors torch.Tensor -- number of datasets x number of trials x number of neurons x time steps
        :param D_exclusive:
        :param D_shared:
        :param kwargs:
        """
        device = kwargs.get('device', 'cpu')
        ell_init = kwargs.get('ell_init', 5.)
        lr = kwargs.get('lr', 5e-2)

        self.n_datasets = len(Y)
        self.N_total = sum([Y[i].shape[1] for i in range(self.n_datasets)])
        self.Y = [Y[i].to(device) for i in range(self.n_datasets)]


        self.T = self.Y[0].shape[-1]
        assert [self.Y[i].shape[-1] == self.T for i in
                range(self.n_datasets)], 'Datasets should be of the same length of recording'

        self.D_shared = D_shared
        self.D_exclusive = D_exclusive

        # initialize inference object for each of "dataset"
        self.vbs = [VariationalBayesInference(self.Y[i], D=self.D_exclusive, **kwargs) for i in range(self.n_datasets)]

        # initialize shared latents and shared weights
        self.shared_latents = Latents(self.D_shared, self.T, device=device, ell=ell_init)
        self.shared_latents.initialize()

        self.shared_weights = [Weights(self.vbs[i].N, self.D_shared, device=device) for i in range(self.n_datasets)]

        # jitter matrices
        self.jitter_weight = (1e-6 * torch.diag_embed(torch.ones(1, self.D_shared).double())).to(device)
        self.jitter_latent = 1e-6 * torch.eye(self.T).double().to(device)

        # M optimizer
        self.optimizer = torch.optim.Adam([
            {
                'params': self.shared_latents._ell, 'lr': lr
            }
        ])

    def F_shared(self):
        return [self.shared_weights[i].mean @ self.shared_latents.mean for i in range(self.n_datasets)]

    def F(self):
        F_shared = self.F_shared()
        return [F_shared[i] + self.vbs[i].F for i in range(self.n_datasets)]

    def success_prob(self):
        F = self.F()
        return [ 1 / (1 + torch.exp(-F[i])) for i in range(self.n_datasets)]

    def dispersion_mean(self):
        return [self.vbs[i].dispersions.mean for i in range(self.n_datasets)]

    def F_i(self, i):
        F_shared_i = self.shared_weights[i].mean @ self.shared_latents.mean
        return F_shared_i + self.vbs[i].F

    def E_F_squared(self):
        # get the E_F_squared of every one
        # add the quadratic terms to them

        E_F_squared_exclusive = [self.vbs[i].E_F_squared() for i in range(self.n_datasets)]
        F_exclusive = [self.vbs[i].F for i in range(self.n_datasets)]
        F_shared = self.F_shared()
        E_F_cross = [F_shared[i] * F_exclusive[i] for i in range(self.n_datasets)]
        E_X_X = self.shared_latents.quadratic_expectation

        result = []
        for i in range(self.n_datasets):
            # E[F_shared^2]
            quad_term = torch.einsum('nd,det->net', self.shared_weights[i].mean, E_X_X)
            quad_term = (quad_term * self.shared_weights[i].mean.unsqueeze(-1)).sum(
                1)  # sum across latent dimensions => N X T
            trace_term = torch.diagonal(torch.einsum('det,nef->ndft', E_X_X, self.shared_weights[i].covariance),
                                        dim1=1, dim2=2).sum(-1)
            E_WXXW = quad_term + trace_term
            # /E[F_shared^2]
            result.append(E_WXXW + 2 * E_F_cross[i] + E_F_squared_exclusive[i])

        return result

    def update_base(self):
        """
        Update base intensities for each datasets
        """
        F_shared = self.F_shared()
        for i in range(self.n_datasets):
            Omega = self.vbs[i].Omega()
            precision = 1 / self.vbs[i].base.prior_covariance + Omega.sum(1, keepdims=True)
            updated_covariance = 1 / precision
            # latent functions from both shared and exclusive latents
            F_total = self.vbs[i].F_XW + F_shared[i]
            updated_mean = updated_covariance * (self.vbs[i].Kappa - F_total * Omega).sum(1, keepdims=True)

            self.vbs[i].base.update(updated_mean, updated_covariance)
            self.vbs[i].update_base_precision()

    def latents_quadratic_expectation(self):
        E_X_X_T = []
        X_shared = self.shared_latents.mean
        X_shared_cov = torch.diagonal(self.shared_latents.posterior_K, dim1=-2, dim2=-1)

        for i in range(self.n_datasets):
            mean = self.vbs[i].latents.mean
            mean = torch.vstack([mean, X_shared])
            # concatenate with X_shared

            # E[X_{d1,t}] E[X_{d2,t}] # D X D X T
            mean_product_terms = torch.einsum('ij,kj-> ikj', mean, mean)

            # Var[X_d] along the diagonal in the DXD dimensions
            diag_cov = torch.diagonal(self.vbs[i].latents.posterior_K, dim1=-2, dim2=-1)
            # concategnate with X_shared_cov
            diag_cov = torch.vstack([diag_cov, X_shared_cov])

            variances = torch.diag_embed(
                diag_cov.transpose(-1, -2)).transpose(-1, -3)

            E_X_X_T.append(mean_product_terms + variances)

        return E_X_X_T

    def update_weight(self):
        """
        Update both shared and exclusive weights together
        :return:
        """
        # D x D latent mean product E[X] E[X]^T
        E_X_X_T = self.latents_quadratic_expectation()

        for i in range(self.n_datasets):
            Omega = self.vbs[i].Omega()
            OtherEffects = self.vbs[i].F_base

            covariance_expectation_term = torch.einsum(
                'det,nt->nde',
                E_X_X_T[i],
                Omega
            )

            # TODO: can be optimized  -- saving only the diagonal instead of the matrix
            prior_precision = torch.diag_embed(
                torch.hstack([
                    torch.diagonal(self.vbs[i].weights.prior_precision, dim1=-2, dim2=-1),
                    torch.diagonal(self.shared_weights[i].prior_precision, dim1=-2, dim2=-1)
                ]))

            D_total = prior_precision.shape[-1]
            jitter_weight = (1e-6 * torch.diag_embed(torch.ones(1, D_total).double())).to(prior_precision.device)
            updated_covariance = torch.linalg.inv(
                prior_precision + covariance_expectation_term + jitter_weight)

            latents_i = torch.vstack([self.vbs[i].latents.mean, self.shared_latents.mean])

            mean_term = (self.vbs[i].Kappa - OtherEffects * Omega) @ latents_i.T.double()
            updated_mean = (updated_covariance @ mean_term.unsqueeze(-1)).squeeze(-1)

            # slice the result into shared mean and covariance
            shared_mean, shared_covariance = (updated_mean[:, -self.D_shared:],
                                              updated_covariance[:, -self.D_shared:, -self.D_shared:])

            exclusive_mean, exclusive_covariance = (updated_mean[:, :self.vbs[i].D],
                                                    updated_covariance[:, :self.vbs[i].D, :self.vbs[i].D])

            # update both shared & exclusive weights
            self.shared_weights[i].update(shared_mean, shared_covariance)
            self.vbs[i].weights.update(exclusive_mean, exclusive_covariance)

            # update prior precision -- ARD Gamma variables
            alpha = 1e-5 + 0.5 * self.vbs[i].N
            beta = 1e-5 + 0.5 * self.vbs[i].weights.quadratic_expectation_diag().sum(0)
            self.vbs[i].weights.update_prior_precision(torch.diag_embed(alpha / beta).unsqueeze(0))

        # update shared weight precision -- for all  weights across datasets
        self.update_weight_precision()

    def update_exclusive_weights(self):
        F_shared = self.F_shared()
        for i in range(self.n_datasets):
            Omega = self.vbs[i].Omega()
            OtherEffects = self.vbs[i].base.mean + F_shared[i]

            # D x D latent mean product E[X] E[X]^T
            E_X_X_T = self.vbs[i].latents.quadratic_expectation

            covariance_expectation_term = torch.einsum(
                'det,nt->nde',
                E_X_X_T,
                Omega
            )

            updated_covariance = torch.linalg.inv(
                self.vbs[i].weights.prior_precision + covariance_expectation_term + self.vbs[i].jitter_weight)

            # N x D
            mean_term = (self.vbs[i].Kappa - OtherEffects * Omega) @ self.vbs[i].latents.mean.T.double()
            updated_mean = (updated_covariance @ mean_term.unsqueeze(-1)).squeeze(-1)

            self.vbs[i].weights.update(updated_mean, updated_covariance)
            self.vbs[i].update_weight_precision()

    def update_exclusive_latents(self):
        F_shared = self.F_shared()
        for i in range(self.n_datasets):

            E_W_W = self.vbs[i].weights.quadratic_expectation_diag()
            Omega = self.vbs[i].Omega()

            Kappa = self.vbs[i].Kappa
            prior_K_inv = self.vbs[i].latents.prior_K_inv().detach()

            for d in range(self.vbs[i].D):
                tmp_d = (E_W_W[:, d][..., None] * Omega).sum(0)

                updated_cov = torch.linalg.inv(prior_K_inv[d].detach() + torch.diag(tmp_d) + self.vbs[i].jitter_latent)

                # TODO: add assertions
                assert torch.all(torch.diag(
                    updated_cov) > 0.).item(), 'WARN:  diagonal values of the covariance are not all non-negative'

                total_effect = self.F_i(i) + F_shared[i]
                other_effects = total_effect - self.vbs[i].weights.mean[:, d:d + 1] @ self.vbs[i].latents.mean[d:d + 1,
                                                                                      :]

                mean_term_d = ((Kappa - other_effects * Omega).T @ self.vbs[i].weights.mean[:, d:d + 1]).squeeze(-1)

                updated_mean = updated_cov @ mean_term_d

                # zero-center latents
                updated_mean = updated_mean - updated_mean.mean()
                self.vbs[i].latents.update(updated_mean, updated_cov, d)

            loss = - self.vbs[i].latents()
            self.vbs[i].optimizer.zero_grad()
            loss.backward()
            self.vbs[i].optimizer.step()
            self.vbs[i].latents.prior_K(update=True)
            self.vbs[i].latents.prior_K_inv(update=True)

    def update_shared_latents(self):
        E_W_W = [self.shared_weights[i].quadratic_expectation_diag() for i in range(self.n_datasets)]
        Omega = [self.vbs[i].Omega() for i in range(self.n_datasets)]
        Kappa = [self.vbs[i].Kappa for i in range(self.n_datasets)]

        prior_K_inv = self.shared_latents.prior_K_inv().detach()

        F_exclusive = [self.vbs[i].F for i in range(self.n_datasets)]

        for d in range(self.D_shared):
            tmp_d = torch.stack([(E_W_W[i][:, d][..., None] * Omega[i]).sum(0) for i in range(self.n_datasets)]).sum(0)

            updated_cov = torch.linalg.inv(prior_K_inv[d].detach() + torch.diag(tmp_d) + self.jitter_latent)

            F_shared = self.F_shared()  # recompute in every loop since it changes at the end of loop
            total_effect = [F_exclusive[i] + F_shared[i] for i in range(self.n_datasets)]
            other_effects = [
                total_effect[i] - self.shared_weights[i].mean[:, d:d + 1] @ self.shared_latents.mean[d: d + 1, :] for i
                in
                range(self.n_datasets)]

            mean_term_d = [
                ((Kappa[i] - other_effects[i] * Omega[i]).T @ self.shared_weights[i].mean[:, d:d + 1]).squeeze(-1) for i
                in
                range(self.n_datasets)]

            # sum up values
            mean_term_d = torch.stack(mean_term_d).sum(0)
            updated_mean = updated_cov @ mean_term_d

            # zero-center latents
            updated_mean = updated_mean - updated_mean.mean()

            self.shared_latents.update(updated_mean, updated_cov, d)

        loss = - self.shared_latents()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.shared_latents.prior_K(update=True)
        self.shared_latents.prior_K_inv(update=True)

    def update_shared_weights(self):
        # D x D latent mean product E[X] E[X]^T
        E_X_X_T = self.shared_latents.quadratic_expectation

        for i in range(self.n_datasets):
            Omega = self.vbs[i].Omega()
            OtherEffects = self.vbs[i].F

            covariance_expectation_term = torch.einsum(
                'det,nt->nde',
                E_X_X_T,
                Omega
            )

            updated_covariance = torch.linalg.inv(
                self.shared_weights[i].prior_precision + covariance_expectation_term + self.jitter_weight)

            # N_i x D
            mean_term = (self.vbs[i].Kappa - OtherEffects * Omega) @ self.shared_latents.mean.T.double()
            updated_mean = (updated_covariance @ mean_term.unsqueeze(-1)).squeeze(-1)

            self.shared_weights[i].update(updated_mean, updated_covariance)

        self.update_weight_precision()

    def update_weight_precision(self):
        """
        Update precision values for shared weights
        """
        alpha = 1e-5 + 0.5 * self.N_total
        beta = 1e-5 + 0.5 * torch.vstack([
            self.shared_weights[i].quadratic_expectation_diag().sum(0) for i in range(self.n_datasets)
        ]).sum(0)
        updated_precision = torch.diag_embed(alpha / beta).unsqueeze(0)

        for i in range(self.n_datasets):
            self.shared_weights[i].update_prior_precision(updated_precision)

    def update_augmented_vars(self):
        """
        Update augmented variables -- Omega, Xi, Tau
        """
        E_F_squared = self.E_F_squared()
        for i in range(self.n_datasets):
            B = self.vbs[i].B

            r_mean = self.vbs[i].dispersions.mean
            r_var = self.vbs[i].dispersions.variance

            E_r_squared = torch.square(r_mean) + r_var

            # update omega -- update dist for each trial
            for m in range(self.vbs[i].M):
                self.vbs[i].omega[m].update(B[m], torch.sqrt(E_F_squared[i]))
                self.vbs[i].tau[m].update(B[m])

            self.vbs[i].xi.update(torch.sqrt(E_r_squared))

    def update_dispersion(self):
        F = self.F()
        for i in range(self.n_datasets):
            Xi = self.vbs[i].xi.mean

            updated_p = self.vbs[i].M * self.T * torch.ones_like(self.vbs[i].dispersions.p)
            updated_a = self.vbs[i].M * self.T * Xi
            updated_b = self.vbs[i].M * self.T * (np.euler_gamma - np.log(2.))
            E_log_tau = torch.stack([torch.digamma(self.vbs[i].tau[m].alpha) for m in range(self.vbs[i].M)]).sum(0)

            F_i = F[i]

            F_i = torch.clamp(F_i, min=-self.vbs[i].F_clamp, max=self.vbs[i].F_clamp)
            F_i = F_i * self.vbs[i].M

            updated_b += (-0.5 * F_i + E_log_tau).sum(-1)

            self.vbs[i].dispersions.update(updated_p, updated_a, updated_b)
