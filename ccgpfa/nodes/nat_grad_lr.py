import torch
class NatGradLearningRate:
    """
    Ranganath et. al , An Adaptive Learning Rate for Stochastic Variational Inference
    https://proceedings.mlr.press/v28/ranganath13.pdf
    """

    def __init__(self, grad, tau_init = 10, rho_init=0.5):
        self.rho_init = rho_init
        self.tau_init = tau_init
        self.tau = self.tau_init # windown size

        self.rho = rho_init

        # E[g_t]
        self.grad_mean = grad

        # E[g_t^T g_t] - quadratic expectation of gradient
        self.grad_quad_exp = torch.square(grad).sum() + 0. # 0 - variance of a single sample
        self.t = 1


    def step(self, new_grad):
        self.t += 1
        # update the mean and variances
        self.grad_mean = (1 - 1/self.tau) * self.grad_mean + (1/self.tau) * new_grad
        self.grad_quad_exp = (1 - 1/self.tau) * self.grad_quad_exp + (1/self.tau) * torch.square(new_grad).sum()



        assert self.grad_mean.shape == new_grad.shape
        if self.t > 10:

            # update the tau and rho
            self.rho = torch.square(self.grad_mean).sum() / self.grad_quad_exp
            self.tau = self.tau * (1 - self.rho) + 1
            if self.rho == 1.:

                self.rho = self.rho_init
                self.tau= self.tau_init
                # import ipdb; ipdb.set_trace()


        return self.rho

