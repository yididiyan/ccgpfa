import os

import torch
import mgplvm as mgp
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import nbinom, binom
from ccgpfa.utils import detach, hinton
from experiments.utils import *



def run(Y,
        ell_init=5.,
        llk='negbinom',
        n_dim=5,
        n_iter=250,
        output_dir=None,
        kinematics=False,
        cs=None,
        test_data=None,
        n_train_trials=1):
    non_zero_trials = Y.sum(-1).sum(0) > 0
    test_data = test_data[:, non_zero_trials, :]
    Y = Y[:, non_zero_trials, :] # filter out neurons with zero spikes

    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    ntrials, n, T = Y.shape # Y should have shape: [number of trials (here 1) x neurons x time points]
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

    logliks = []
    r_vals = []

    def F_and_R(mod):
        F = mod.obs.q_mu.data @ (mod.lat_dist.lat_mu.transpose(-1, -2) * mod.obs.dim_scale.T.unsqueeze(-1))
        F = detach(mod.obs.likelihood.c[..., None] * F + mod.obs.likelihood.d[..., None])
        R = detach(mod.obs.likelihood.total_count)  # number of failures
        return F, R

    prob = None
    def cb(mod, i, loss):
        # F is link to log odds via sigmoid transformation  --- "success" function
        F, R = F_and_R(mod)
        r_vals.append(R)


        if i % 50 == 0:
            prob = 1 / ( 1 + np.exp(-F))
            logliks.append(nbinom.logpmf(Y, R[None, ..., None], 1 - prob).sum(0).sum(-1))


    train_ps = mgp.crossval.training_params(max_steps=max_steps, n_mc=n_mc, lrate=5e-2, batch_size=80, callback=cb)
    print('fitting', n, 'neurons and', T, 'time bins for', max_steps, 'iterations')

    mgp.crossval.train_model(mod, data, train_ps)
    # predicted firing rate
    F, R = F_and_R(mod)
    rate = np.exp(F) * R[None, ..., None]

    weights = mod.obs.dim_scale.T[None, ...] * mod.obs.q_mu


    plt.plot(rate[0][0])
    plt.bar(np.arange(test_data.shape[-1]), test_data.mean(0)[0])
    plt.show()



    # plot them figures
    fig, (ax1, ax2, ax3) = plt.subplots(3, 2, constrained_layout=True, figsize=(10, 10))
    hinton(detach(mod.obs.q_mu[0]).T[..., :10], ax=ax1[0])
    ax1[0].set_title('Weights')

    logliks = np.array(logliks).squeeze() / n_train_trials
    r_vals = np.array(r_vals)

    for n in range(Y.shape[1]):
        ax1[1].plot(logliks[:, n])
        ax2[0].plot(r_vals[:, n])

    ax1[1].set_xlabel('iterations')
    ax1[1].set_title(f'llk (per neuron)')

    ax2[0].set_title('Dispersion values')

    ax2[1].plot(logliks.sum(-1))
    ax2[1].set_title(f'Total loglikelihood -- {logliks.sum(-1)[-1]}')

    ax3[0].plot(detach(mod.lat_dist.lat_mu[0]))
    ax3[0].set_title('Latents - inferred')

    F = mod.obs.q_mu.data @ (mod.lat_dist.lat_mu.transpose(-1, -2) * mod.obs.dim_scale.T.unsqueeze(-1))
    F = detach(mod.obs.likelihood.c[..., None] * F + mod.obs.likelihood.d[..., None])

    n_timepoints = int(F.shape[-1] / n_train_trials)
    # F_reshaped = F.reshape(n_train_trials, F.shape[1], n_timepoints)
    # F_mean = F_reshaped.mean(0, keepdims=True)
    # prob = 1 / (1 + np.exp(-F_mean))

    prob_mean = ( ((1 / (1 + np.exp(-F)))
                 .reshape(
                    n_train_trials,
                    F.shape[1],
                    n_timepoints).mean(0, keepdims=True)))

    # ax3[1].hist(Y[0].flatten(), label='true', alpha=0.5)
    # # compute test llk
    test_llk_mean, test_llk_std = None, None
    import ipdb; ipdb.set_trace()
    if test_data is not None:
        test_llk = nbinom.logpmf(test_data, r_vals[-1][None, ..., None], 1 - prob_mean)
        test_llk_mean, test_llk_std = test_llk.mean(0).sum(), test_llk.std(0).sum()
        print(f'Test logliklihood {test_llk_mean} \pm {test_llk_std}')

    # # ax3[1].hist(nbinom.rvs(r_vals[-1][..., None], 1 - prob[0]).flatten(), label='inferred', alpha=0.5)
    # ax3[1].legend()
    # ax3[1].set_title('Count distributions')

    fig.suptitle(f'Results - {llk} - R_max - init lengthscale - {ell_init}')

    if output_dir:
        # save results
        fig.savefig(f'{output_dir}/summary.png')

    fig.show()





if __name__ == '__main__':
    import argparse
    from pathlib import Path
    from ccgpfa.inference.vb_neg_binomial_multi_trial import generate

    parser = argparse.ArgumentParser(description='Allen Data experiments')
    # seed to initialize the data & var dists
    # parser.add_argument('--dataset', action='store', type=str,
    #                     default='/media/yido/additional/research/Gitlab/gpfa/icml_experiments/synthetic/data_2000_20.pkl')
    # parser.add_argument('--output_dir', action='store', type=str,
    #                     default='/media/yido/additional/research/Gitlab/gpfa/icml_experiments/allen_data')
    parser.add_argument('--dataset', action='store', type=str,
                        default='/media/yido/additional/research/Github/allen_data/icml-data/drifting_gratings_75_repeats/data_15_45.0_0.1_V1.pkl')
    parser.add_argument('--output', action='store', type=str,
                        default='/media/yido/additional/research/Gitlab/gpfa/icml_experiments/allen_data/drifting_gratings_75_repeats')
    parser.add_argument('--n-iter', action='store', type=int, default=1000)  # latent dimensions
    parser.add_argument('--n-dim', action='store', type=int, default=10) # latent dimensions
    parser.add_argument('--ell', action='store', type=float, default=5.)  # initial lengthscale values
    parser.add_argument('--test-size', action='store', type=int, default=25)
    parser.add_argument('--kinematics', action='store', type=bool, default=False)  # initial lengthscale values


    parser.add_argument('--llk', action='store', type=str, default='negbinom')

    args = parser.parse_args()

    # output directory
    output_dir = f'{args.output}/{Path(args.dataset).stem}/bgpfa_independent_trials_{args.llk}'
    os.makedirs(output_dir, exist_ok=True)


    data = load_data(args.dataset)

    # shuffle trials
    rng = np.random.default_rng(0)
    rng.shuffle(data)

    n_trials = data.shape[0]
    n_train_trials = n_trials - args.test_size
    test_data = data[-args.test_size:, ...]
    data = data[:n_trials-args.test_size, ...]

    for i in range(data.shape[0]):
        # for each trial
        vb = run(
            data[i:i+1, ...],
            llk=args.llk,
            ell_init=args.ell,
            n_dim=args.n_dim,
            n_iter=args.n_iter,
            output_dir=output_dir,
            test_data=None,
            n_train_trials=1)


    with open(f'{output_dir}/summary.pkl', 'wb') as f_:
        pickle.dump({
            'r_vals': r_vals,
            'llks': logliks,
            'test_llk_mean': test_llk_mean,
            'test_llk_std': test_llk_std,
            'ell': detach(mod.obs.dim_scale),
            'weights': weights,
            'spike_prob': prob
        }, f_)

