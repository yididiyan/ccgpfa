import os

import torch
import mgplvm as mgp
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import nbinom, binom
from ccgpfa.utils import detach, hinton
from experiments.utils import *
from run_with_behavior_mapping import zero_pad
import time



def load_data_kinematics(f, swapaxes=True):
    with open(f, 'rb') as f_:
        data = pickle.load(f_)

        Y = data['data'].astype(np.float32)
        if swapaxes: Y = Y.swapaxes(-1, -2)

        # trial x T x direction
        hand_vel = data['hand_velocity']
        # hand_vel = np.array([[vel['x'], vel['y']] for vel in data['hand_velocity']]).swapaxes(-1, -2)

    return Y, hand_vel

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
    non_zero_indices = Y.sum(-1).sum(0) > 0
    test_data = test_data[:, non_zero_indices, :] if not test_data is None else None
    Y = Y[:, non_zero_indices, :] # filter out neurons with zero spikes

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
    start_time = time.time()

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
    # helper function to specify training parameters


    train_ps = mgp.crossval.training_params(max_steps=max_steps, n_mc=n_mc, lrate=5e-2, batch_size=32, callback=cb)
    print('fitting', n, 'neurons and', T, 'time bins for', max_steps, 'iterations')

    mgp.crossval.train_model(mod, data, train_ps)
    # predicted firing rate
    F, R = F_and_R(mod)
    rate = np.exp(F) * R[None, ..., None]

    weights = mod.obs.dim_scale.T[None, ...] * mod.obs.q_mu


    plt.plot(rate[0][0])
    # plt.bar(np.arange(test_data.shape[-1]), test_data.mean(0)[0])
    # plt.show()



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
    F_mean = F.mean(0, keepdims=True)

    prob_mean = 1 / (1 + np.exp(-F_mean))
    # ax3[1].hist(Y[0].flatten(), label='true', alpha=0.5)
    # # compute test llk
    elapsed_time = time.time() - start_time
    test_llk, test_llk_mean, test_llk_std = None, None, None

    if test_data is not None:
        test_llk = nbinom.logpmf(test_data, r_vals[-1][None, ..., None], 1 - prob_mean)
        test_llk_mean, test_llk_std = test_llk.mean(0).sum(), test_llk.std(0).sum()
        print(f'Test logliklihood {test_llk_mean} \pm {test_llk_std} --{ elapsed_time }')



    fig.suptitle(f'Results - {llk} - R_max - init lengthscale - {ell_init}')

    if output_dir:
        # save results
        fig.savefig(f'{output_dir}/summary.png')

    fig.show()
    firing_rates = np.array([ zero_pad((len(non_zero_indices), T), rate[i], non_zero_indices) for i in range(len(rate))])
    assert firing_rates.shape == (rate.shape[0], len(non_zero_indices), T)


    return {
        'F': F,
        'r_vals': r_vals,
        'llks': logliks, # mean training loglik
        'test_llk': test_llk,
        'test_llk_mean': test_llk_mean,
        'test_llk_std': test_llk_std,
        'ell': detach(mod.obs.dim_scale),
        'weights': weights,
        'spike_prob': prob_mean,
        'elapsed_time': elapsed_time,
        'firing_rates': firing_rates
    }





if __name__ == '__main__':
    import argparse
    from pathlib import Path
    from ccgpfa.inference.vb_neg_binomial_multi_trial import generate

    parser = argparse.ArgumentParser(description='Allen Data experiments')
    parser.add_argument('--dataset', action='store', type=str,
                        default='')
    parser.add_argument('--output_dir', action='store', type=str,
                        default='')
    parser.add_argument('--n-iter', action='store', type=int, default=2000)  # latent dimensions
    parser.add_argument('--n-dim', action='store', type=int, default=10) # latent dimensions
    parser.add_argument('--ell', action='store', type=float, default=5.)  # initial lengthscale values
    parser.add_argument('--test-size', action='store', type=int, default=0)
    parser.add_argument('--kinematics', action='store', type=bool, default=False)  # initial lengthscale values


    parser.add_argument('--llk', action='store', type=str, default='negbinom')

    args = parser.parse_args()

    # output directory
    output_dir = f'{args.output_dir}/{Path(args.dataset).stem}/bgpfa_{args.llk}'
    os.makedirs(output_dir, exist_ok=True)

    behavior_data = None
    if args.kinematics:
        Y, behavior_data = load_data_kinematics(args.dataset)
    else:
        Y = load_data(args.dataset)

    # shuffle trials
    shuffle_seed = 0
    rng = np.random.default_rng(shuffle_seed)
    rng.shuffle(Y)

    if args.test_size:
        train_data = Y[:-args.test_size, ...]
        test_data = Y[-args.test_size:, ...]

        print(f'Train trials - {train_data.shape[0]}, test trials {test_data.shape[0]}')
    else:
        train_data = Y
        test_data = None



    results = {}

    for type, data, other_data in zip(['train', 'test'], [train_data, test_data], [test_data, train_data]):
        if np.any(data):
            n_train_trials = data.shape[0]
            result = run(data, llk=args.llk, ell_init=args.ell, n_dim=args.n_dim, n_iter=args.n_iter,
                             output_dir=output_dir, test_data=test_data, n_train_trials=n_train_trials)

            results[type] = result

    result['shuffle_seed'] = shuffle_seed

    # save all data
    with open(f'{output_dir}/summary.pkl', 'wb') as f_:
        pickle.dump(results, f_)
