import os

import torch
import mgplvm as mgp
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import nbinom, binom
from ccgpfa.utils import detach, hinton
from experiments.utils import *
import time



def run(Y,
        ell_init=5.,
        llk='negbinom',
        n_dim=5,
        n_iter=250,
        output_dir=None,
        cs=None,
        test_data=None,
        n_train_trials=1, device = 'cuda'):
    # filter out neurons with zero trials
    non_zero_trials = Y.sum(-1).sum(0) > 0
    test_data = test_data[:, non_zero_trials, :]
    Y = Y[:, non_zero_trials, :] # filter out neurons with zero spikes
    print(f' Train data - {Y.shape} test data - {test_data.shape}' )
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    data = torch.tensor(Y).to(device) # put the data on our GPU/CPU
    ts = np.arange(Y.shape[-1]) #much easier to work in units of time bins here
    fit_ts = torch.tensor(ts)[None, None, :].to(device) # put our time points on GPU/CPU

    ### set some parameters for fitting ###
    ell0 = ell_init # initial timescale (in bins) for each dimension. This could be the ~timescale of the behavior of interest (otherwise a few hundred ms is a reasonable default)
    rho = 2 # sets the intial scale of each latent (s_d in Jensen & Kao). rho=1 is a natural choice with Gaussian noise; less obvious with non-Gaussian noise but rho=1-5 works well empirically.
    max_steps = n_iter # number of training iterations
    n_mc = 10 # number of monte carlo samples per iteration



    ### construct the actual model ###
    ntrials, n, T = Y.shape # Y should have shape: [number of trials (here 1) x neurons x time points]
    lik = mgp.likelihoods.NegativeBinomial(n, Y=Y) # we use a negative binomial noise model in this example (recommended for ephys data)
    manif = mgp.manifolds.Euclid(T, n_dim) # our latent variables live in a Euclidean space for bGPFA (see Jensen et al. 2020 for alternatives)
    var_dist = mgp.rdist.GP_circ(manif, T, ntrials, fit_ts, _scale=1, ell = ell0) # circulant variational GP posterior (c.f. Jensen & Kao et al. 2021)
    lprior = mgp.lpriors.Null(manif) # here the prior is defined implicitly in our variational distribution, but if we wanted to fit e.g. Factor analysis this would be a Gaussian prior
    mod = mgp.models.Lvgplvm(n, T, n_dim, ntrials, var_dist, lprior, lik, Y = Y, learn_scale = False, ard = True, rel_scale = rho).to(device) #create bGPFA model with ARD
    start_time = time.time()

    r_vals = []

    def F_and_R(mod):
        F = mod.obs.q_mu.data @ (mod.lat_dist.lat_mu.transpose(-1, -2) * mod.obs.dim_scale.T.unsqueeze(-1))
        F = detach(mod.obs.likelihood.c[..., None] * F + mod.obs.likelihood.d[..., None])
        R = detach(mod.obs.likelihood.total_count)  # number of failures
        prob = 1 / (1 + np.exp(-F))
        return F, R, prob

    prob = None

    def cb(mod, i, loss):
        R = detach(mod.obs.likelihood.total_count)
        r_vals.append(R)


    # helper function to specify training parameters
    train_ps = mgp.crossval.training_params(max_steps=max_steps, n_mc=n_mc, lrate=5e-2, batch_size=32, callback=cb)
    print('fitting', n, 'neurons and', T, 'time bins for', max_steps, 'iterations')

    mgp.crossval.train_model(mod, data, train_ps)

    weights = mod.obs.dim_scale.T[None, ...] * mod.obs.q_mu


    # compute the training logliklilihood
    F, R, prob = F_and_R(mod)
    loglik = nbinom.logpmf(Y, R[None, ..., None], 1 - prob)
    print(loglik.sum(axis=(1,2)).mean(), ' -- mean training loglik ') # mean training loglik



    # plot them figures
    fig, (ax1, ax2, ax3) = plt.subplots(3, 2, constrained_layout=True, figsize=(10, 10))
    hinton(detach(mod.obs.q_mu[0]).T[..., :10], ax=ax1[0])
    ax1[0].set_title('Weights')

    r_vals = np.array(r_vals)


    ax1[1].set_xlabel('iterations')
    ax1[1].set_title(f'llk (per neuron)')

    ax2[0].set_title('Dispersion values')

    ax3[0].plot(detach(mod.lat_dist.lat_mu[0]))
    ax3[0].set_title('Latents - inferred')

    F = mod.obs.q_mu.data @ (mod.lat_dist.lat_mu.transpose(-1, -2) * mod.obs.dim_scale.T.unsqueeze(-1))
    F = detach(mod.obs.likelihood.c[..., None] * F + mod.obs.likelihood.d[..., None])
    F_mean = F.mean(0, keepdims=True)

    prob_mean = 1 / (1 + np.exp(-F_mean))
    elapsed_time = time.time() - start_time
    test_llk_mean, test_llk_std = None, None

    if test_data is not None:
        test_llk = - nbinom.logpmf(test_data, r_vals[-1][None, ..., None], 1 - prob_mean)
        # test_llk_mean, test_llk_std = test_llk.mean(0).sum(), test_llk.std(0).sum()
        test_llk_mean, test_llk_std = test_llk.mean(), test_llk.std() / np.sqrt(test_llk.reshape(-1).shape[0])
        print(f'Test logliklihood {test_llk_mean} \pm {test_llk_std} -- Time taken{ elapsed_time }')


    fig.suptitle(f'Results - {llk} - R_max - init lengthscale - {ell_init}')

    if output_dir:
        # save results
        fig.savefig(f'{output_dir}/summary.png')

    fig.show()
    with open(f'{output_dir}/summary.pkl', 'wb') as f_:
        pickle.dump({
            'F': F,
            'r_vals': r_vals,
            'test_llk': test_llk,
            'test_llk_mean': test_llk_mean,
            'test_llk_std': test_llk_std,
            'ell': detach(mod.obs.dim_scale),
            'weights': weights,
            'spike_prob': prob_mean,
            'elapsed_time': elapsed_time
        }, f_)




if __name__ == '__main__':
    import argparse
    from pathlib import Path
    from ccgpfa.inference.vb_neg_binomial_multi_trial import generate

    parser = argparse.ArgumentParser(description='Allen Data experiments')
    parser.add_argument('--dataset', action='store', type=str,
                        default=None)
    parser.add_argument('--output_dir', action='store', type=str,
                        default=None)
    parser.add_argument('--n-iter', action='store', type=int, default=2000)  # latent dimensions
    parser.add_argument('--n-dim', action='store', type=int, default=10) # latent dimensions
    parser.add_argument('--ell', action='store', type=float, default=5.)  # initial lengthscale values
    parser.add_argument('--test-size', action='store', type=int, default=25)
    parser.add_argument('--shuffle', action='store', type=int, default=0)


    parser.add_argument('--llk', action='store', type=str, default='negbinom')

    args = parser.parse_args()

    # output directory
    output_dir = f'{args.output_dir}/{Path(args.dataset).stem}/bgpfa_{args.llk}'
    os.makedirs(output_dir, exist_ok=True)


    data = load_data(args.dataset)

    # shuffle trials
    rng = np.random.default_rng(args.shuffle)
    rng.shuffle(data)

    n_trials = data.shape[0]
    n_train_trials = n_trials - args.test_size
    test_data = data[-args.test_size:, ...]
    data = data[:n_trials-args.test_size, ...]
    vb = run(data, llk=args.llk, ell_init=args.ell, n_dim=args.n_dim, n_iter=args.n_iter,
             output_dir=output_dir, test_data=test_data, n_train_trials=n_train_trials)