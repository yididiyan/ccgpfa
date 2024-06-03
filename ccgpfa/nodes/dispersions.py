import numpy as np
import torch
from ccgpfa.utils import detach


class Dispersions:
    """
    Negative binomial dispersion parameters
    """

    def __init__(self, N, device=None, n_mc_samples=1, max_r=5.):
        """

        :param N: number of neurons
        :param device: computation device (CPU/CUDA)
        :param n_mc_samples: number of Monte Carlo samples
        :param max_r: upper bound on the r value
        """
        self.N = N
        self.n_mc_samples = n_mc_samples
        self.max_r = max_r

        self.p = torch.ones((N,)).double().to(device)
        self.a = torch.ones((N,)).double().to(device)
        self.b = torch.ones((N,)).double().to(device)

        # initialize the means and variances of the variables
        self._mean = 1. * torch.ones((N,)).double().to(device)
        # set initial vairance to be the max R
        self._var = self.max_r * torch.ones((N,)).double().to(device)

        #
        # self.simp = Simpson()
        # self.integration_domain = torch.Tensor([[0, self.max_r]]).double().to(device)

    @property
    def mean(self):
        return self._mean

    @property
    def variance(self):
        return self._var

    def sample(self, p, a, b):
        def compute_tau():
            """
            He et al. .2022 - Appendix C
            """

            positive_b_indices = b > 0
            negative_b_indices = b < 0

            tau = torch.sqrt(0.25 + 2 * a * p * (1 / (b ** 2)))

            tau[positive_b_indices] += 0.5
            tau[negative_b_indices] -= 0.5

            return tau

        def matching_gamma():
            return p, compute_tau() * torch.abs(b) - b  # two params of gamma

        def sample_ptn():
            """
            Single sample from PTN(p,a,b)
            """
            shape, rate = matching_gamma()
            tau = compute_tau()

            ##
            shape = shape.repeat(self.n_mc_samples)
            rate = rate.repeat(self.n_mc_samples)
            tau = tau.repeat(self.n_mc_samples)
            a_ = a.repeat(self.n_mc_samples)
            b_ = b.repeat(self.n_mc_samples)

            sample = torch.ones_like(shape) * torch.nan  # initialize as nan

            while True:
                indices = torch.isnan(sample)
                u = torch.rand(sample[indices].shape).to(a.device)  # sample from uniform dist
                dist = torch.distributions.gamma.Gamma(shape[indices], rate[indices])
                gamma_sample = dist.sample()
                accept_indices = u < torch.exp(
                    -a_[indices] * (gamma_sample - tau[indices] * torch.abs(b_[indices]) / (2 * a_[indices])) ** 2)
                gamma_sample[~accept_indices] = torch.nan  # mask rejected samples
                sample[indices] = gamma_sample

                if not torch.isnan(sample).any():
                    return sample

        return sample_ptn()

    def _compute_mean(self):
        samples = self.sample(self.p, self.a, self.b).reshape(self.n_mc_samples, self.N)
        self._mean = samples.mean(0)
        self._var = torch.square(torch.nan_to_num(samples.std(0)))

        # bounding the means
        self._mean[self._mean > self.max_r] = self.max_r

    def _compute_mean_2(self):
        """
        approximate the mean with the mode of the distribution
        :return:
        """

        self._mean = (-self.b - torch.sqrt(torch.square(self.b) + 8 * self.a * (self.p - 1))) / (-4 * self.a)
        self._var = 0 * self._var

        # bounding the means
        self._mean[self._mean > self.max_r] = self.max_r

    def exp_r_squared(self):
        return torch.square(self._mean) + self._var

    def update(self, p, a, b):
        self.p = p
        self.a = a
        self.b = b

        self._compute_mean()


class DispersionsTruncated(Dispersions):

    def sample(self, p, a, b):


        def compute_tau():
            """
            He et al. .2022 - Appendix C
            """

            positive_b_indices = b > 0
            negative_b_indices = b < 0

            tau = torch.sqrt(0.25 + 2 * a * p * (1 / (b ** 2)))

            tau[positive_b_indices] += 0.5
            tau[negative_b_indices] -= 0.5

            return tau

        def matching_gamma():
            return p, compute_tau() * torch.abs(b) - b  # two params of gamma

        def sample_ptn():
            """
            Single sample from PTN(p,a,b)
            """
            shape, rate = matching_gamma()
            tau = compute_tau()

            ##
            from scipy.stats import gamma

            # cdf evaluate at max r
            dist = torch.distributions.Gamma(shape, rate)
            if type(self.max_r) == float:
                self.max_r = torch.Tensor([self.max_r])

            cdf_max_r = dist.cdf(self.max_r)

            shape = shape.repeat(self.n_mc_samples)
            rate = rate.repeat(self.n_mc_samples)
            cdf_max_r = cdf_max_r.repeat(self.n_mc_samples)

            tau = tau.repeat(self.n_mc_samples)
            a_ = a.repeat(self.n_mc_samples)
            b_ = b.repeat(self.n_mc_samples)

            sample = torch.ones_like(shape) * torch.nan  # initialize as nan

            while True:
                indices = torch.isnan(sample)
                u = torch.rand(sample[indices].shape).to(a.device)  # sample from uniform dist

                # sample the un truncated Gamma distributions for values in (0, self.max_r]
                u_2 = torch.rand(sample[indices].shape).to(a.device)
                u_2 = u_2 * cdf_max_r[indices]

                # gamma samples that satisfy the first condition
                gamma_samples = gamma.ppf(detach(u_2), detach(shape[indices]), scale=1/detach(rate[indices]))
                gamma_samples = torch.Tensor(gamma_samples).double().to(a.device)

                # accept_indices = u < (torch.exp(
                #     -a_[indices] * (
                #             gamma_samples
                #             - tau[indices] * torch.abs(b_[indices])
                #             / (2 * a_[indices])) ** 2)) / cdf_max_r[indices]
                accept_indices = u < 1.

                gamma_samples[~accept_indices] = torch.nan  # mask rejected samples
                sample[indices] = gamma_samples

                if not torch.isnan(sample).any():
                    return sample

        return sample_ptn()

    def _compute_mean(self):
        samples = self.sample(self.p, self.a, self.b).reshape(self.n_mc_samples, self.N)
        self._mean = samples.mean(0)
        self._var = torch.square(torch.nan_to_num(samples.std(0)))



class DispersionsNumericalIntegration(Dispersions):
    """
    Compute mean of distribution using numerical integration
    """

    def _compute_mean(self):
        def ptn_truncated_mean(x, p, a, b):
            """
            x - set of points to evaluate the function on N x 1
            A total of M PTN variables described by parameters below
            p - M x 1
            a - M x 1
            b - M x 1

            returns an M x 1 --- mean of the dists
            """

            # compute the inner term
            inner = (p - 1) @ torch.log(x) - a @ (x ** 2) + b @ x
            probs = torch.softmax(inner, dim=1)  # change to probabilities

            return torch.sum(x * probs, -1)


        x = torch.arange(0, self.max_r, 0.1).double()[None, ...]

        self._mean = ptn_truncated_mean(x, self.p[..., None], self.a[..., None], self.b[..., None])
        self._var = torch.zeros_like(self._mean)



if __name__ == '__main__':
    import matplotlib.pyplot as plt

    N = 20
    dispersions = Dispersions(N, n_mc_samples=20)

    for i in range(N):
        dispersions.p[i] = 8 * (i + 1)
        dispersions.a[i] = 10 * (i + 1)
        dispersions.b[i] = -20 * (i + 1)

    dispersions._compute_mean()

    mean_vals = dispersions.mean.cpu().numpy()
    var_vals = dispersions.variance.cpu().numpy()

    plt.figure()
    plt.plot(mean_vals)
    plt.title('Mean')
    plt.show()

    plt.figure()
    plt.plot(dispersions.variance.cpu().numpy())
    plt.title('Variance')
    plt.show()

    plt.figure()
    plt.errorbar(range(N), mean_vals, yerr=var_vals)
    plt.show()
