import torch

from ccgpfa.utils import safe_inv_softplus, safe_softplus


class Latents(torch.nn.Module):

    def __init__(self, D, T, device=None, ell=20, ell_bound=5., generator: torch.Generator = None, scale=.1, *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
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
        mean_product_terms = torch.einsum('ij,kj-> ikj', self.mean, self.mean)

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
                -self.distance.double() / (2 * torch.square(self.ell.double().unsqueeze(-1)))).double()

        return self._prior_K

    def prior_K_inv(self, update=False):
        if update:
            self._prior_K_inv = torch.linalg.inv(self._prior_K.double() + self.jitter * self.I)

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

        """
        if d is not None:
            self._mean[d:d + 1, :] = mean
            self._posterior_K[d:d + 1, :, :] = K
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


class SparseLatents(torch.nn.Module):

    def __init__(self, D, T, M, device=None, ell=20, ell_bound=5., generator: torch.Generator = None, scale=.1, *args,
                 **kwargs):
        """

        :param D: number of dimensions
        :param T: timesteps
        :param device: device
        :param ell: init lengthscale
        :param ell_bound: sampling bound
        :param generator:
        :param scale:
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        self.D = D
        self.T = T
        self.M = M
        self.device = device

        self.ts_mm = torch.Tensor(torch.linspace(0, self.T, self.M)).double()
        self.ts_mm = torch.nn.Parameter(self.ts_mm, requires_grad=True)
        self.ts_tt = torch.Tensor(torch.arange(0, self.T)).double()
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
        self.update_distances()


        # jitter value and eye matrix
        self.jitter = 1e-6
        self.I_mm = torch.diag_embed(torch.ones(self.D, self.ts_mm.shape[-1])).to(device)

    def update_distances(self):
        # squared distance term can be pre-calculated
        self.distance_mm = torch.square(self.ts_mm.unsqueeze(-1) - self.ts_mm.unsqueeze(-2))
        # self.distance_mm = self.distance_mm.repeat(self.D, 1, 1).to(device)
        self.distance_mm = self.distance_mm.expand(self.D, -1, -1).to(self.device).detach()

        self.distance_mt = torch.square(self.ts_mm.unsqueeze(-1) - self.ts_tt.unsqueeze(-2))
        # self.distance_mt = self.distance_mt.repeat(self.D, 1, 1).to(device)
        self.distance_mt = self.distance_mt.expand(self.D, -1, -1).to(self.device).detach()

        self.distance_tt = torch.square(self.ts_tt.unsqueeze(-1) - self.ts_tt.unsqueeze(-2))
        # self.distance_tt = self.distance_tt.repeat(self.D, 1, 1).to(device)
        self.distance_tt = self.distance_tt.expand(self.D, -1, -1).to(self.device)

    def initialize(self):
        """
        Initialize prior covariance matrix
        and posterior mean and covariance
        """
        # prior covariance terms
        self._prior_K_mm = self.prior_K_mm(update=True) # COV(U, U)
        self._prior_K_inv_mm = self.prior_K_mm_inv(update=True) # COV(U, U)^-1
        self._prior_K_mt = self.prior_K_mt(update=True)  # Cov(U, X)

        # prior covariance of the mm matrix
        self._prior_K_tt = self.prior_K_tt(update=True)  # Cov(X, X)
        self._prior_K_mm_inv_K_mt = self.prior_K_mm_inv_K_mt(update=True)

        # posterior mean & covariance
        # posterior mean & covariance
        self._posterior_K_mm = self._prior_K_mm.detach().clone()
        # sampling mean values as initializations - zero mean results in "zero" updates
        K_half_mm = torch.linalg.cholesky(
            self._posterior_K_mm
            + (self.jitter * self.I_mm)
        )

        # initialize the posterior to help kick-start the CAVI updates
        self._U_mean = (K_half_mm @ torch.randn(self.D, self.ts_mm.shape[0], generator=self.generator).double().to(self.device).unsqueeze(
            -1) * self.scale).squeeze(-1)
        K_mm_inv_K_mt = (self._prior_K_mm_inv @ self._prior_K_mt).detach()
        self._mean = (K_mm_inv_K_mt.transpose(-1, -2) @ self._U_mean.unsqueeze(-1)).squeeze(-1)


        # initialize posterior K_tt
        self._posterior_K_tt = (self._prior_K_tt - K_mm_inv_K_mt.transpose(-1,
                                                                           -2) @ self._prior_K_mm_inv @ self._prior_K_mt).detach().clone()

    @property
    def mean(self):
        return self._mean

    @property
    def U_mean(self):
        """
        E[U]
        Returns:

        """
        return self._U_mean

    @property
    def quadratic_expectation(self):
        """
        E[X X^T] = Cov(.,.) + E[X]^2 -- the quadratic term
        Returns: DXDXT -- quadratic expression

        """
        # E[X_{d1,t}] E[X_{d2,t}] # D X D X T
        mean_product_terms = torch.einsum('ij,kj-> ikj', self.mean, self.mean)

        # Var[X_d] along the diagonal in the DXD dimensions
        variances = torch.diag_embed(
            torch.diagonal(self.posterior_K_tt, dim1=-2, dim2=-1).transpose(-1, -2)).transpose(-1, -3)

        return mean_product_terms + variances


    @property
    def ell(self):
        return safe_softplus(self._ell)

    """
    Prior covariance matrices and their inverses 
    """

    def prior_K_tt(self, update=False, sample_indices=None):
        if update:
            self._prior_K_tt = torch.exp(-self.distance_tt / (2 * torch.square(self.ell.detach().unsqueeze(-1))))

        return self._prior_K_tt

    def prior_K_mm(self, update=False):
        """
        prior covariance of inducing pionts
        Returns:
        """
        if update:
            self._prior_K_mm = torch.exp(-self.distance_mm / (2 * torch.square(self.ell.detach().unsqueeze(-1))))

        return self._prior_K_mm

    def prior_K_mt(self, update=False, sample_indices=None):
        """
        K_mt -- covariance matrix
        """
        if update:
            self._prior_K_mt = torch.exp(-self.distance_mt / (2 * torch.square(self.ell.detach().unsqueeze(-1))))

        return self._prior_K_mt

    def prior_K_mm_inv(self, update=False):
        if update:
            self._prior_K_mm_inv = torch.linalg.inv(self._prior_K_mm + self.jitter * self.I_mm)

        return self._prior_K_mm_inv
    @property
    def posterior_K_mm(self):
        # self.prior_K_inv()
        return self._posterior_K_mm

    @property
    def posterior_K_tt(self):
        # self.prior_K_inv()
        return self._posterior_K_tt

    def prior_K_mm_inv_K_mt(self, update=False, sample_indices=None):
        if update:
            self._prior_K_mm_inv_K_mt = self._prior_K_mm_inv @ self._prior_K_mt
        return self._prior_K_mm_inv_K_mt


    def update(self, mean, K, d=None, sample_indices=None):
        """
        Update posterior mean and covariance
        Args:
            mean: posterior mean -- n_samples X D X M
            K: posterior covariance -- n_samples X D X M X M

            Eq(4) in https://arxiv.org/pdf/2012.13962.pdf
            q(X) = \int p(X|U) q(U) dU

        """
        if d is not None:
            # update the U's
            self._U_mean[d, :] = mean
            self._posterior_K_mm[d, :, :] = K

            # update the X's
            # Mean -> (K_mm^{-1} K_mt)^T @  U_mean
            # K_mm^{-1} K_mt ---> prior matrices
            K_mm_inv_K_mt = self.prior_K_mm_inv_K_mt()[d, :, :].detach()

            # Remark: I shouldn't be doing this everytime I update U; it's expensive and doesn't help


            self._mean[d, :] = (K_mm_inv_K_mt.transpose(-1, -2) @  mean.unsqueeze(-1)).squeeze(-1)
            # K_tt - K_tm K_mm^-1( K_mm - S_mm) K_mm^-1 K_mt
            # prior - prior prior^{-1} (prior - posterior) prior^{-1} prior^{-1}

            self._posterior_K_tt[d, :] = self.prior_K_tt()[d, :] - K_mm_inv_K_mt.transpose(-1, -2) @ (self._prior_K_mm[d, :, :] - K) @ K_mm_inv_K_mt


        else:
            raise NotImplementedError('Updates should be component wise ')

    def forward(self):
        jitter = self.jitter * self.I_mm
        mean = self.U_mean
        distance_mm = torch.square(self.ts_mm.unsqueeze(-1) - self.ts_mm.unsqueeze(-2))
        distance_mm = distance_mm.expand(self.D, -1, -1).to(self.device).detach()
        prior_K_mm = torch.exp(-distance_mm / (2 * torch.square(self.ell.unsqueeze(-1))))
        prior_K_mm_inv = torch.linalg.inv(prior_K_mm + self.jitter * self.I_mm)


        trace_term = torch.diagonal(self.posterior_K_mm @ prior_K_mm_inv, dim1=-2, dim2=-1).sum(-1)
        mean_term = (mean.unsqueeze(-2).double() @ prior_K_mm_inv @ mean.unsqueeze(-1).double()).sum(-1).sum(-1)
        det_term = torch.logdet(prior_K_mm + jitter)  # penalty term
        return -0.5 * torch.nansum(trace_term + mean_term + det_term)

    def update_sample_indices(self, indices):
        pass



class SparseStochasticLatent(torch.nn.Module):

    def __init__(self, D, T, M, device=None, ell=20, ell_bound=5., generator: torch.Generator = None, scale=.1, sample_indices=None, *args,
                 **kwargs):
        """

        :param D: number of dimensions
        :param T: timesteps
        :param device: device
        :param ell: init lengthscale
        :param ell_bound: sampling bound
        :param generator:
        :param scale:
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        self.D = D
        self.T = T
        self.M = M
        self.device = device

        # to accommodate boundary time steps well, we consider evenly selected time steps [-1, T+1]
        self.ts_mm = torch.Tensor(torch.linspace(-1, self.T+1, self.M)).double()
        # self.ts_mm = torch.nn.Parameter(self.ts_mm, requires_grad=True)
        self.ts_tt = torch.Tensor(torch.arange(0, self.T)).double()
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


        # jitter value and eye matrix
        self.jitter = 1e-6
        self.I_mm = torch.diag_embed(torch.ones(self.D, self.ts_mm.shape[-1])).to(device)
        self.sample_indices = sample_indices

        # squared distance term can be pre-calculated
        self.distance_mm = torch.square(self.ts_mm.unsqueeze(-1) - self.ts_mm.unsqueeze(-2))
        # self.distance_mm = self.distance_mm.repeat(self.D, 1, 1).to(device)
        self.distance_mm = self.distance_mm.expand(self.D, -1, -1).to(self.device).detach()


        self.update_distances()


    def update_sample_indices(self, sample_indices):
        """
        Method that needs to be called everytime we change our sample indices ---> once every iteration
        """
        self.sample_indices = sample_indices

        # update distances b/n inducing points and original points
        self.update_distances()

        # mean and variance functions
        self.prior_K_tt(update=True)
        self.prior_K_mt(update=True)
        self.prior_K_mm_inv_K_mt(update=True)
        self.posterior_K_tt(update=True)  # also posterior mean is updated


    def update_distances(self):
        self.ts_tt = torch.Tensor(self.sample_indices).double()

        self.distance_mt = torch.square(self.ts_mm.unsqueeze(-1) - self.ts_tt.unsqueeze(-2))
        # self.distance_mt = self.distance_mt.repeat(self.D, 1, 1).to(device)
        self.distance_mt = self.distance_mt.expand(self.D, -1, -1).to(self.device).detach()

        self.distance_tt = torch.square(self.ts_tt.unsqueeze(-1) - self.ts_tt.unsqueeze(-2))
        # self.distance_tt = self.distance_tt.repeat(self.D, 1, 1).to(device)
        self.distance_tt = self.distance_tt.expand(self.D, -1, -1).to(self.device)

    def initialize(self):
        """
        Initialize prior covariance matrix
        and posterior mean and covariance
        """
        # prior covariance terms
        self._prior_K_mm = self.prior_K_mm(update=True) # COV(U, U)
        self._prior_K_inv_mm = self.prior_K_mm_inv(update=True) # COV(U, U)^-1
        # self._prior_K_mt = self.prior_K_mt(update=True)  # Cov(U, X)



        # prior covariances
        self._prior_K_tt = None
        self._prior_K_mm_inv_K_mt = None
        self._prior_K_mt = None

        # posterior mean & covariance
        self._posterior_K_mm = self._prior_K_mm.detach().clone()
        # sampling mean values as initializations - zero mean results in "zero" updates
        K_half_mm = torch.linalg.cholesky(
            self._posterior_K_mm
            + (self.jitter * self.I_mm)
        )

        # initialize the posterior to help kick-start the CAVI updates
        self._U_mean = (K_half_mm @ torch.randn(self.D, self.ts_mm.shape[0], generator=self.generator).double().to(self.device).unsqueeze(
            -1) * self.scale).squeeze(-1)

        # posterior in natural parameterization
        self._posterior_eta_2 = -0.5 * self._prior_K_mm_inv.detach().clone()
        self._posterior_eta_1 = (-2 * self._posterior_eta_2 @ self._U_mean.unsqueeze(-1)).squeeze(-1)



        # initialize posterior K_tt
        self._posterior_K_tt = None

    @property
    def mean(self):
        return self._mean

    @property
    def U_mean(self):
        """
        E[U]
        Returns:

        """
        return self._U_mean

    @property
    def quadratic_expectation(self):
        """
        E[X X^T] = Cov(.,.) + E[X]^2 -- the quadratic term
        Returns: DXDXT -- quadratic expression

        """
        mean_ = self.mean
        posterior_K_tt_ = self.posterior_K_tt()
        # E[X_{d1,t}] E[X_{d2,t}] # D X D X T
        mean_product_terms = torch.einsum('ij,kj-> ikj', mean_, mean_)

        # Var[X_d] along the diagonal in the DXD dimensions
        variances = torch.diag_embed(
            torch.diagonal(posterior_K_tt_, dim1=-2, dim2=-1).transpose(-1, -2)).transpose(-1, -3)

        return mean_product_terms + variances


    @property
    def ell(self):
        return safe_softplus(self._ell)

    """
    Prior covariance matrices and their inverses 
    """

    def prior_K_tt(self, update=False, sample_indices=None):
        if update:
            self._prior_K_tt = torch.exp(-self.distance_tt / (2 * torch.square(self.ell.detach().unsqueeze(-1))))

        return self._prior_K_tt

    def prior_K_mm(self, update=False):
        """
        prior covariance of inducing pionts
        Returns:
        """
        if update:
            self._prior_K_mm = torch.exp(-self.distance_mm / (2 * torch.square(self.ell.detach().unsqueeze(-1))))

        return self._prior_K_mm

    def prior_K_mt(self, update=False):
        """
        K_mt -- covariance matrix
        """
        if update:
            self._prior_K_mt = torch.exp(-self.distance_mt / (2 * torch.square(self.ell.detach().unsqueeze(-1))))

        return self._prior_K_mt

    def prior_K_mm_inv(self, update=False):
        if update:
            self._prior_K_mm_inv = torch.linalg.inv(self._prior_K_mm + self.jitter * self.I_mm)

        return self._prior_K_mm_inv
    @property
    def posterior_K_mm(self):
        # self.prior_K_inv()
        return self._posterior_K_mm

    def posterior_K_tt(self, update=False):
        if update:
            K_mm_inv_K_mt = (self._prior_K_mm_inv @ self._prior_K_mt).detach()
            K = self._posterior_K_mm
            self._mean = (K_mm_inv_K_mt.transpose(-1, -2) @ self.U_mean.unsqueeze(-1)).squeeze(-1)

            self._posterior_K_tt = self.prior_K_tt() - K_mm_inv_K_mt.transpose(-1, -2) @ (
                    self._prior_K_mm - K
            ) @ K_mm_inv_K_mt

        return self._posterior_K_tt

    def prior_K_mm_inv_K_mt(self, update=False):
        if update:
            self._prior_K_mm_inv_K_mt = self._prior_K_mm_inv @ self._prior_K_mt
        return self._prior_K_mm_inv_K_mt




    def update(self, mean, K, d=None):
        """
        Update posterior mean and covariance
        Args:
            mean: posterior mean -- n_samples X D X M
            K: posterior covariance -- n_samples X D X M X M

            Eq(4) in https://arxiv.org/pdf/2012.13962.pdf
            q(X) = \int p(X|U) q(U) dU

        """

        raise NotImplementedError("Use natural graident update instead")

    def update_natural(self, eta_1, eta_2, d, step_size=1., zero_center=True):
        assert eta_1.shape == self._posterior_eta_1[d, ...].shape
        assert eta_2.shape == self._posterior_eta_2[d, ...].shape

        self._posterior_eta_1[d, :] = step_size * eta_1 + (1 - step_size) * self._posterior_eta_1[d,:]
        self._posterior_eta_2[d, ...] = step_size * eta_2 + (1 - step_size) * self._posterior_eta_2[d,...]


        # transform the natural params
        K = torch.linalg.inv(
            self._posterior_eta_2[d, ...] * -2 + self.jitter * self.I_mm[d, ...])  # add jitter
        assert torch.all(torch.diagonal(K) > 0), 'diagonal with negative entries'
        mean = (K @ self._posterior_eta_1[d, :].unsqueeze(-1)).squeeze(-1)

        # update the U's
        if zero_center:
            mean = mean - mean.mean()
        self._U_mean[d, :] = mean
        self._posterior_K_mm[d, :, :] = K

        # update the X's
        # Mean -> (K_mm^{-1} K_mt)^T @  U_mean
        # K_mm^{-1} K_mt ---> prior matrices
        K_mm_inv_K_mt = self.prior_K_mm_inv_K_mt()[d, :, :].detach()

        # project points to mini-batch points
        self._mean[d, :] = (K_mm_inv_K_mt.transpose(-1, -2) @  mean.unsqueeze(-1)).squeeze(-1)
        # K_tt - K_tm K_mm^-1( K_mm - S_mm) K_mm^-1 K_mt
        # prior - prior prior^{-1} (prior - posterior) prior^{-1} prior^{-1}

        self._posterior_K_tt[d, :] = self.prior_K_tt()[d, :] - K_mm_inv_K_mt.transpose(-1, -2) @ (
                self._prior_K_mm[d, :, :] - K
        ) @ K_mm_inv_K_mt

        assert torch.all(torch.diagonal(self._posterior_K_tt[d]) > 0)

        # import matplotlib.pyplot as plt
        # plt.imshow(K.detach().cpu().numpy())
        # plt.show()
    def forward(self):
        jitter = self.jitter * self.I_mm
        mean = self.U_mean
        distance_mm = torch.square(self.ts_mm.unsqueeze(-1) - self.ts_mm.unsqueeze(-2))
        # currently not learning locations .detach() here removes it from the graph
        distance_mm = distance_mm.expand(self.D, -1, -1).to(self.device).detach()
        prior_K_mm = torch.exp(-distance_mm / (2 * torch.square(self.ell.unsqueeze(-1))))
        prior_K_mm_inv = torch.linalg.inv(prior_K_mm + self.jitter * self.I_mm)


        trace_term = torch.diagonal(self.posterior_K_mm @ prior_K_mm_inv, dim1=-2, dim2=-1).sum(-1)
        mean_term = (mean.unsqueeze(-2).double() @ prior_K_mm_inv @ mean.unsqueeze(-1).double()).sum(-1).sum(-1)
        det_term = torch.logdet(prior_K_mm + jitter)  # penalty term
        return -0.5 * torch.nansum(trace_term + mean_term + det_term)





if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from ccgpfa.utils import detach

    L = torch.Tensor([0.5, .5])
    D, T = 2, 100

    latents = Latents(D, T)
    latents.initialize()
    print(latents.distance.shape)
    assert latents.distance.shape[0] == D and latents.distance.shape[1] == T
    # assert latents._L.requires_grad
    K = latents._prior_K[0]
    plt.imshow(K.detach().cpu().numpy())
    plt.show()


    sparse_latent = SparseLatents(D, T, 5)
    sparse_latent.initialize()

    K_mm = sparse_latent._prior_K_mm[0]
    plt.imshow(K_mm.detach().cpu().numpy())
    plt.show()

    K_mt = sparse_latent._prior_K_mt[0]
    plt.imshow(K_mt.detach().cpu().numpy())
    plt.show()

    plt.plot(detach(sparse_latent.mean[0]))
    plt.show()

