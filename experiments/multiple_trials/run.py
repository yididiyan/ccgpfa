import os
import time

import matplotlib.pyplot as plt
import numpy as np

from ccgpfa.utils import detach
from ccgpfa.inference.vb_neg_binomial_multi_trial import VariationalBayesInference as NegBinomVB
from ccgpfa.inference.vb_neg_binomial_multi_trial import SparseVariationalInference as SparseNegBinomVB
from ccgpfa.inference.stochastic_vb_neg_binomial_multi_trials import StochasticSparseVariationalInference as StochasticNegBinomVB
from ccgpfa.inference.stochastic_vb_binomial_multi_trials import StochasticSparseVariationalInference as StochasticBinomVB
from ccgpfa.inference.vb_binomial_multi_trial import VariationalBayesInference as BinomVB
from ccgpfa.inference.vb_binomial_multi_trial import SparseVariationalInference as SparseBinomVB

from experiments.utils import *

import torch
plt.rcParams.update({'font.size': 22})

def run(Y,
        ell_init=5.,
        llk='negbinom',
        max_r=1000.,
        n_dim=5,
        n_iter=250,
        F_clamp=1.,
        output_dir=None,
        kinematics=False,
        cs=None,
        no_base=False,
        test_data=None,
        sparse=False,
        n_inducing=None):
    # non_zero_indices = Y.sum(-1).sum(0) > 0
    # Y = Y[:, non_zero_indices , :]
    # test_data = test_data[:, non_zero_indices, :] if not test_data is None else None
    N = Y.shape[1]
    print(f'Using f clamp {F_clamp} -- no base - {no_base}')
    vb = None
    if llk == 'negbinom':
        if n_inducing:
            # import ipdb; ipdb.set_trace()
            # vb = SparseNegBinomVB(torch.Tensor(Y), n_dim, M=n_inducing, ell_init=ell_init, n_mc_samples=10, max_r=max_r,
            #            F_clamp=F_clamp, device='cuda', tied_trials=False)
            vb = StochasticNegBinomVB(torch.Tensor(Y),n_dim, M=n_inducing, tied_samples=False,
                                                      batch_size=50, F_clamp=F_clamp)
        else:
            vb = NegBinomVB(torch.Tensor(Y), n_dim, ell_init=ell_init, n_mc_samples=10, max_r=max_r,
                                   F_clamp=F_clamp, device='cuda', tied_trials=False)
    elif llk == 'binom':

        if n_inducing:
            vb = SparseBinomVB(torch.Tensor(Y), n_dim, M=n_inducing, ell_init=ell_init, device='cuda', tied_trials=False)
        else:
            vb = BinomVB(torch.Tensor(Y), n_dim, ell_init=ell_init, device='cuda', tied_trials=False)


    # # initialize base intensity as zero
    vb.base.mean = 0. * vb.base.mean
    vb.base.covariance = 0. * vb.base.covariance
    start_time = time.time()

    vb, logliks, test_logliks, r_vals = train_model(vb,  n_iter, llk, test_data=test_data, no_base=no_base)

    title = f'Results - {llk} - R_max {max_r} - Clip value(F) {F_clamp}\n init lengthscale - {ell_init}'
    fig = plot_results(vb, logliks, r_vals, title)



    # Recalculating llk
    prob = detach(vb.success_prob)[None, ...]
    if llk == 'negbinom':
        tmp = nbinom.logpmf(detach(vb.Y), detach(vb.dispersions.mean)[None, ..., None], 1 - prob)
    else:
        n = detach(vb.n)[None, ..., None]
        tmp = binom.logpmf(detach(vb.Y), n, prob)
    llk_mean, llk_std = tmp.mean(0).sum(), tmp.std(0).sum()
    test_llk_mean, test_llk_std = None, None

    elapsed_time = time.time() - start_time
    print(f'Elapased time {elapsed_time}')

    if test_data is not None:


        if llk == 'negbinom':
            tmp = nbinom.logpmf(test_data, detach(vb.dispersions.mean)[None, ..., None], 1 - prob)
        else:
            n = detach(vb.n)[None, ..., None]
            tmp = binom.logpmf(test_data, n, prob)

        test_llks_ = np.nan_to_num(tmp, neginf=0.)
        test_llk_mean, test_llk_std = test_llks_.mean(0).sum(), test_llks_.std(0).sum()

    print(llk_mean, test_llk_mean, llk_std, test_llk_std)
    # end recalculating llk


    if output_dir:
        # save results
        fig.savefig(f'{output_dir}/summary.png')
        ell = detach(vb.latents.ell)

        # individual plots
        plt.figure()
        plot_weights(detach(vb.weights.mean).T[:, :10])
        plt.title('Loading Weights')
        plt.savefig(f'{output_dir}/weights.pdf')

        plt.figure()
        plot_loglik(logliks.sum(-1), test_logliks.sum(-1))
        plt.savefig(f'{output_dir}/loglik.pdf')

        # orthonormalize
        plt.figure()
        W_orth, X_orth = orthonormalize(detach(vb.weights.mean), detach(vb.latents.mean))
        plot_latents(X_orth)
        plt.savefig(f'{output_dir}/orth_latents.pdf')
        plt.show()

        plt.figure()
        psth(detach(vb.Y), detach(vb.firing_rate))
        plt.savefig(f'{output_dir}/firing_rate.png')
        plt.savefig(f'{output_dir}/firing_rate.pdf', dpi=300)

        with open(f'{output_dir}/summary.pkl', 'wb') as f_:
            pickle.dump({
                'r_vals': r_vals,
                'llks': logliks,
                'test_llks': test_logliks, # during training
                'test_llks_': test_llks_,
                'test_llk_mean': test_llk_mean,
                'test_llk_std': test_llk_std,
                'llk_mean': llk_mean,
                'llk_std': llk_std,
                'weights': detach(vb.weights.mean),
                'latents': detach(vb.latents.mean),
                'n_iter': n_iter,
                'ell': ell,
                'firing_rates': detach(vb.firing_rate),
                'elapsed_time': elapsed_time
            }, f_)



    return vb

def psth(Y, firing_rates, neuron_idx=10):
    """
    Plot firing rates with PSTH
    :param Y:
    :param firing_rates:
    :return:
    """
    Y_mean = Y.mean(0)
    plt.title('Mean Firing Rate')
    plt.ylabel('Spikes/bin')
    plt.xlabel('Time')
    plt.xticks([])
    for pos in ['right', 'top', ]:
        plt.gca().spines[pos].set_visible(False)
    for i in range(10):
        # plt.plot(firing_rates[i], linestyle='--', linewidth=2, color='tab:blue')
        plt.plot(firing_rates[i], linewidth=2)
    # plt.bar(np.arange(Y.shape[-1]), Y_mean[neuron_idx, :], alpha=0.5, color='gray')
    plt.tight_layout()

def plot_latents(latents):

    plt.title('Latents')
    plt.xlabel('Time')
    plt.xticks([])
    for pos in ['right', 'top', ]:
        plt.gca().spines[pos].set_visible(False)
    plt.plot(latents.T, linewidth=1.)
    plt.tight_layout()


def plot_weights(weight):
    hinton(weight)
    plt.xlabel('Latent dimensions')
    plt.ylabel('Neurons')
    for pos in ['right', 'top', 'bottom', 'left']:
        plt.gca().spines[pos].set_visible(False)
    plt.tight_layout()


def plot_loglik(loglik, test_loglik):
    plt.title('Loglikelihood')
    plt.xlabel('Iterations')
    for pos in ['right', 'top', ]:
        plt.gca().spines[pos].set_visible(False)
    plt.plot(loglik, linewidth=1., label='held-in')
    plt.plot(test_loglik, linewidth=1., label='held-out')
    plt.legend()
    plt.tight_layout()

if __name__ == '__main__':
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description='Allen Data experiments')
    # seed to initialize the data & var dists
    # parser.add_argument('--dataset', action='store', type=str,
    #                     default='/media/yido/additional/research/Gitlab/gpfa/icml_experiments/synthetic/data_2000_20.pkl')

    parser.add_argument('--dataset', action='store', type=str,
                        default='/media/yido/additional/research/Github/allen_data/icml-data/drifting_gratings_75_repeats/data_15_45.0_0.8_V1.pkl')
    parser.add_argument('--output_dir', action='store', type=str,
                        default='/media/yido/additional/research/Gitlab/gpfa/icml_experiments/allen_data/drifting_gratings_75_repeats')
    # parser.add_argument('--output_dir', action='store', type=str,
    #                     default='/media/yido/additional/research/Gitlab/gpfa/icml_experiments/allen_data')
    parser.add_argument('--test-size', action='store', type=int, default=25)
    parser.add_argument('--kinematics', action='store', type=bool, default=False)
    parser.add_argument('--n-iter', action='store', type=int, default=50)  # latent dimensions
    parser.add_argument('--n-dim', action='store', type=int, default=10) # latent dimensions
    parser.add_argument('--ell', action='store', type=float, default=5)  # initial lengthscale values
    parser.add_argument('--r-max', action='store', type=float, default=10.) # max value for R
    parser.add_argument('--f-clamp', action='store', type=float, default=1.)
    parser.add_argument('--llk', action='store', type=str, default='negbinom')
    parser.add_argument('--no-base', action='store', type=bool, default=False)
    parser.add_argument('--single', action='store', type=bool, default=False)

    # number of inducing point every "n-inducing" points
    parser.add_argument('--n-inducing', action='store', type=int, default=None)


    args = parser.parse_args()

    # output directory
    output_dir = f'{args.output_dir}/{Path(args.dataset).stem}/{args.llk}___f_clamp{args.f_clamp}_n_inducing{args.n_inducing}'
    os.makedirs(output_dir, exist_ok=True)


    if args.kinematics:
        file = args.dataset
        file = '/media/yido/additional/research/Gitlab/gpfa/icml_experiments/data/Doherty_example_1500_3000.pkl'
        Y, locs, cs = load_kinematics_data(file)
        print(Y.shape, locs.shape, cs)

        vb, gen_data = run(Y, llk=args.llk, ell_init=args.ell, n_dim=args.n_dim,
                 n_iter=args.n_iter, output_dir=output_dir, kinematics=True, cs=cs, max_r=args.r_max)

        with open(f'/media/yido/additional/research/Gitlab/gpfa/icml_experiments/data/synthetic/{Path(file).name}', 'wb') as f_:
            pickle.dump({
                'data': gen_data
            }, f_)

    else:
        Y = load_data(args.dataset)

        # shuffle trials
        rng = np.random.default_rng(0)
        rng.shuffle(Y)

        # Y = Y[:10, ...]
        train_data = Y[:-args.test_size, ...]
        test_data = Y[-args.test_size:, ...]
        print(f'Train trials - {train_data.shape[0]}, test trials {test_data.shape[0]}')


        from scipy.ndimage import gaussian_filter1d

        plt.figure(figsize=(4, 4))
        filtered = gaussian_filter1d(train_data.mean(0), 3)

        plt.imshow(filtered, cmap='afmhot')
        plt.xlabel('Time')
        plt.ylabel('Neurons')
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(f'{output_dir}/neural-activity.pdf', dpi=500)
        plt.show()

        vb = run(train_data,
                 llk=args.llk,
                 ell_init=args.ell,
                 n_dim=args.n_dim,
                 F_clamp=args.f_clamp,
                 n_iter=args.n_iter, output_dir=output_dir, max_r=args.r_max, no_base=args.no_base, test_data=test_data, n_inducing=args.n_inducing)
