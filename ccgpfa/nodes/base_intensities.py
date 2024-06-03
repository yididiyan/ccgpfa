import torch


class BaseIntensities:
    """
    Nodes to capture base activity of neurons
    """

    def __init__(self, N, device=None):
        """
        :param N neurons
        """
        self.N = N
        self.prior_mean = torch.zeros((N, 1)).double().to(device)
        self.prior_covariance = torch.ones((1, 1)).double().to(device)  # one for all neurons

        self.covariance = torch.ones((N, 1)).double().to(device)
        self.mean = torch.zeros((N, 1)).double().to(device)

        # Transform to natural parameters
        self._posterior_eta_2 = -0.5 * (1/self.covariance)
        self._posterior_eta_1 = (-2 * self._posterior_eta_2 * self.mean)

        # natural parameters for the Gamma precision params
        self._prior_alpha = 1e-5
        self._prior_beta = 1e-5

        self.precision_eta_1 = 1e-5 - 1
        self.precision_eta_2 = - 1e-5 * torch.ones((1, 1)).double().to(device)

        self.prior_covariance = (self.precision_eta_1 + 1) / (-self.precision_eta_2)



    @property
    def quadratic_expectation(self):
        """
        Computes the quadratic expectation of the variables
        E[bias bias^T] = Cov(., .) + E[bias]^2
        """
        return self.covariance + self.mean ** 2

    def update(self, mean, covariance):
        assert self.mean.shape == mean.shape
        assert self.covariance.shape == covariance.shape
        self.mean = mean
        self.covariance = covariance


    def update_natural(self, eta_1, eta_2, step_size=None):

        """
        Update using natural gradients
        """
        assert eta_1.shape == self._posterior_eta_1.shape
        assert eta_2.shape == self._posterior_eta_2.shape

        self._posterior_eta_1 = step_size * eta_1 + (1 - step_size) * self._posterior_eta_1
        self._posterior_eta_2 = step_size * eta_2 + (1 - step_size) * self._posterior_eta_2

        # Transform to mean and covariance
        self.covariance = - 0.5 * 1 / self._posterior_eta_2
        self.mean = self.covariance * self._posterior_eta_1
