import torch

from ccgpfa.utils import safe_inv_softplus, safe_softplus


class CoupledLatents(torch.nn.Module):

    def __init__(self, n_cond, D, T, device=None, ell=20, ell_bound=5., generator: torch.Generator = None, scale=.1, *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.n_conditions = n_cond
        self.D = D
        self.T = T
        self.device = device

        self.ts = torch.Tensor(torch.arange(self.T))[None, :]
        self.generator = generator
        self.scale = scale

        # initialize length scale
        if ell is None:
            _ell = torch.ones(self.D,
                              1).to(device) * (torch.max(self.ts) - torch.min(self.ts)).cpu() / 20
        else:
            if type(ell) in [float, int]:
                _ell = ell_bound * (torch.rand(self.D, 1, generator=generator) - 0.5) + torch.ones(self.D, 1) * ell
            else:
                _ell = ell
        _ell = torch.abs(_ell)
        self._ell = torch.nn.Parameter(data=safe_inv_softplus(_ell.to(device)), requires_grad=True)

        # time steps, and differences
        # squared distance term can be pre-calculated
        self.distance = torch.square(self.ts.unsqueeze(-1) - self.ts.unsqueeze(-2))
        self.distance = self.distance.repeat(self.D, 1, 1).to(device)

        # jitter value and eye matrix
        self.jitter = 1e-6
        self.I = torch.diag_embed(torch.ones(self.D, self.T)).to(device)

    def initialize(self):
        """
        Initialize prior covariance matrix
        and posterior mean and covariance
        """
        # prior covariance terms
        self._prior_K = self.prior_K(update=True)
        self._prior_K_inv = self.prior_K_inv(update=True)

        # posterior mean & covariance
        # posterior mean & covariance
        self._posterior_K = self._prior_K.detach().clone()
        # sampling mean values as initializations - zero mean results in "zero" updates
        K_half = torch.linalg.cholesky(
            self._posterior_K
            + (self.jitter * self.I)
        )

        # initialize the posterior to help kick-start the CAVI updates
        self._mean = (K_half @ torch.randn(self.D, self.T, generator=self.generator).double().to(self.device).unsqueeze(
            -1) * self.scale).squeeze(-1)

    @property
    def mean(self):
        """
        E[X]
        Returns:

        """
        return self._mean

    @property
    def quadratic_expectation(self):
        """
        E[X X^T] = Cov(.,.) + E[X]^2 -- the quadratic term
        Returns: DXDXT -- quadratic expression

        """
        # E[X_{d1,t}] E[X_{d2,t}] # D X D X T
        mean_product_terms = torch.einsum('bij,bkj-> bikj', self.mean, self.mean)

        # Var[X_d] along the diagonal in the DXD dimensions
        variances = torch.diag_embed(
            torch.diagonal(self.posterior_K, dim1=-2, dim2=-1).transpose(-1, -2)).transpose(-1, -3)

        return mean_product_terms + variances

    def quadratic_expectation_diagonal(self):
        return torch.square(self.mean) + torch.diagonal(self.posterior_K, dim1=-2, dim2=-1)

    # torch.einsum('bij,bkj-> bik', self.latent_nodes.mean, self.latent_nodes.mean), torch.diag_embed(
    # torch.diagonal(self.latent_nodes.posterior_K, dim1=-2, dim2=-1).sum(-1))

    @property
    def ell(self):
        return safe_softplus(self._ell)

    def prior_K(self, update=False):
        """
        Full prior covariance
        Returns:
        """
        if update:
            self._prior_K = torch.exp(
                -self.distance.double() / (2 * torch.square(self.ell.double().unsqueeze(-1)))).double().expand(
                (self.n_conditions, -1, -1, -1 ))

        return self._prior_K

    def prior_K_inv(self, update=False):
        if update:
            self._prior_K_inv = torch.linalg.inv(self._prior_K.double() + self.jitter * self.I).expand(
                (self.n_conditions, -1, -1, -1))

        return self._prior_K_inv

    @property
    def posterior_K(self):
        # self.prior_K_inv()
        return self._posterior_K

    def update(self, mean, K, d=None):
        """
        Update posterior mean and covariance
        Args:
            mean: posterior mean
            K: posterior covariance
            cond: condition index

        """
        if d is not None:
            self._mean[:, d, :] = mean
            self._posterior_K[:, d, :, :] = K
        else:
            self._mean = mean
            self._posterior_K = K

    def kl(self):
        jitter = self.jitter * self.I
        prior_K = self.prior_K().detach()
        prior_L = torch.linalg.cholesky(prior_K + jitter)
        prior_L_inv = torch.linalg.inv(prior_L)
        prior_K_inv = prior_L_inv.transpose(-1, -2) @ prior_L_inv

        prior_K = prior_L @ prior_L.transpose(-1, -2)
        posterior_K = self.posterior_K
        posterior_mean = self.mean.double()  # since the other values are also double

        TrTerm = torch.diagonal(prior_K_inv @ posterior_K, dim1=-2, dim2=-1).sum(-1)
        MeanTerm = (posterior_mean.unsqueeze(-2) @ prior_K_inv @ posterior_mean.unsqueeze(-1)).sum(-1).sum(-1)
        LogDetTerm = torch.logdet(prior_K + jitter) - torch.logdet(posterior_K + jitter)

        kl = 0.5 * (TrTerm + LogDetTerm + MeanTerm - self.T)

        if (kl < 0.).any():
            kl = torch.sqrt(torch.square(kl))
            # print('warning: KL < 0. ')

        return kl

    def forward(self):

        trace_term = torch.diagonal(self._posterior_K @ self._prior_K_inv, dim1=-2, dim2=-1).sum(-1)
        mean_term = (self._mean.unsqueeze(-2).double() @ self._prior_K_inv @ self._mean.unsqueeze(-1).double()).sum(
            -1).sum(-1)
        det_term = torch.logdet(self._prior_K + self.jitter * self.I)  # penalty term

        return -0.5 * (trace_term + mean_term + det_term).sum()


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    L = torch.Tensor([0.5, .5])
    n_cond, D, T = 13, 2, 100

    latents = CoupledLatents(n_cond, D, T)
    latents.initialize()


    print(latents())
    print(latents.distance.shape)
    assert latents.distance.shape[0] == D and latents.distance.shape[1] == T
    # assert latents._L.requires_grad
    K = latents._prior_K[0]
    plt.imshow(K[0].detach().cpu().numpy())
    plt.show()

