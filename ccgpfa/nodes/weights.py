import torch


class Weights:
    """
    Loadings Weights
    """

    def __init__(self, N, D, init_weight=None, device=None):
        """

        :param N: number of neurons
        :param D: number of latent dimensions
        :param device: computation device
        """
        # hyperparameters for Gamma ARD parameter
        self.hyper_alpha = 1e-5
        self.hyper_beta = 1e-5

        self.prior_mean = torch.zeros((N, D)).double().to(device)

        self.prior_precision = (self.hyper_alpha / self.hyper_beta) * torch.diag_embed(
                torch.ones(1, D)).double().to(device)

        self.covariance = torch.diag_embed(torch.ones(N, D)).double().to(device)
        self.mean = torch.randn((N, D)).double().to(device)

        if init_weight is not None:
            self.mean = torch.Tensor(init_weight).double().to(device)

        # Transform to natural parameters
        self._posterior_eta_2 = -0.5 * torch.diag_embed(torch.ones(N, D)).double().to(device)
        self._posterior_eta_1 = (-2 * self._posterior_eta_2 @ self.mean.unsqueeze(-1)).squeeze(-1)

        self.jitter_weight = (1e-6 * torch.diag_embed(torch.ones(1, D).double())).to(device)

    def quadratic_expectation_diag(self):
        return torch.diagonal(self.covariance, dim1=1, dim2=2) + torch.square(self.mean)

    def update(self, mean, covariance):
        assert self.mean.shape == mean.shape
        assert self.covariance.shape == covariance.shape
        self.mean = mean
        self.covariance = covariance

    def update_prior_precision(self, precision):
        """
        update E[tau_d]
        """
        assert self.prior_precision.shape == precision.shape
        assert torch.all(torch.diag(precision[0] ) > 0.) , "negative precision values "
        self.prior_precision = precision

    def update_natural(self, eta_1, eta_2, step_size=1.):
        """
        update using natural graident
        """
        assert eta_1.shape == self._posterior_eta_1.shape
        assert eta_2.shape == self._posterior_eta_2.shape

        # update the natural parameters
        self._posterior_eta_1 = step_size * eta_1 + (1 - step_size) * self._posterior_eta_1
        self._posterior_eta_2 = step_size * eta_2 + (1 - step_size) * self._posterior_eta_2


        # Transform to mean and covariance
        self.covariance = -0.5 * torch.linalg.inv(self._posterior_eta_2 + self.jitter_weight)
        assert torch.all(torch.diagonal(self.covariance, dim1=1, dim2=2) > 0.), 'negative items in a weight diagonal'
        self.mean = (self.covariance @ self._posterior_eta_1.unsqueeze(-1)).squeeze(-1)




if __name__ == '__main__':
    weights = Weights(10, 2)
    print(weights.covariance)
    print(weights.quadratic_expectation_diag())
