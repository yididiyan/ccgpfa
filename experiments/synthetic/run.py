import os
import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

from ccgpfa.nodes.latents import SparseLatents, SparseStochasticLatent
from ccgpfa.utils import detach
from ccgpfa.inference.vb_neg_binomial_multi_trial import VariationalBayesInference as NegBinomVB
from ccgpfa.inference.vb_neg_binomial_multi_trial import SparseVariationalInference as SparseNegBinomVB
from ccgpfa.inference.stochastic_vb_neg_binomial_multi_trials import StochasticSparseVariationalInference as StochasticNegBinomVB
from ccgpfa.inference.stochastic_vb_binomial_multi_trials import StochasticSparseVariationalInference as StochasticBinomVB
from ccgpfa.inference.vb_binomial_multi_trial import VariationalBayesInference as BinomVB
from ccgpfa.inference.vb_binomial_multi_trial import SparseVariationalInference as SparseBinomVB

from experiments.utils import *
from experiments.multiple_trials.run import psth,plot_results, plot_loglik, plot_weights, plot_latents

import torch
plt.rcParams.update({'font.size': 22})


def zero_pad(original_size, source_array, filter_):
    """

    :param original_size: size
    :param source_array: array to use as source
    :param filter_: binary vector
    :return:
    """
    assert sum(filter_) == source_array.shape[0]

    arr = np.zeros(original_size)
    print(arr.shape, source_array.shape)
    arr[filter_, :] = source_array

    return arr


def run(Y,
        ell_init=5.,
        llk='negbinom',
        max_r=1000.,
        n_dim=5,
        n_iter=250,
        F_clamp=1.,
        output_dir=None,
        cs=None,
        no_base=False,
        test_data=None,
        sparse=False,
        n_inducing=None,
        batch_size=25,
        lr=0.25):
    non_zero_indices = Y.sum(-1).sum(0) > 0
    Y = Y[:, non_zero_indices , :]
    test_data = test_data[:, non_zero_indices, :] if not test_data is None else None
    N = Y.shape[1]
    print(f'Using f clamp {F_clamp} -- no base - {no_base} ')
    vb = None
    if llk == 'negbinom':
        if n_inducing:
            print(f'Neg Binomial GPFA -- inducing points {n_inducing}')
            if sparse:
                vb = SparseNegBinomVB(torch.Tensor(Y), n_dim, M=n_inducing, ell_init=ell_init, n_mc_samples=10, max_r=max_r,
                           F_clamp=F_clamp, device='cuda', tied_trials=False, zero_centered=not no_base)
            else:
                vb = StochasticNegBinomVB(torch.Tensor(Y), n_dim, M=n_inducing, tied_samples=False, device='cuda', max_r=max_r,
                                      batch_size=batch_size, F_clamp=F_clamp, zero_centered=not no_base, n_mc_samples=10, ngd_lr=lr)
        else:
            vb = NegBinomVB(torch.Tensor(Y), n_dim, ell_init=ell_init, n_mc_samples=10, max_r=max_r,
                            F_clamp=F_clamp, device='cuda', tied_trials=False, zero_centered=not no_base)
    elif llk == 'binom':
        if n_inducing:
            print(f'Binomial GPFA -- inducing points {n_inducing}')
            if sparse:
                vb = SparseBinomVB(torch.Tensor(Y), n_dim, M=n_inducing, ell_init=ell_init, device='cuda',
                                   tied_trials=False, zero_centered=not no_base)
            else:
                vb = StochasticBinomVB(torch.Tensor(Y),n_dim, M=n_inducing, tied_samples=False, device='cuda',
                                                          batch_size=batch_size, n_mc_samples=10, zero_centered=not no_base, ngd_lr=lr)

        else:
            vb = BinomVB(torch.Tensor(Y), n_dim, ell_init=ell_init, device='cuda', tied_trials=False, zero_centered=not no_base)

    # # initialize base intensity as zero
    vb.base.mean = 0. * vb.base.mean
    vb.base.covariance = 0. * vb.base.covariance
    start_time = time.time()

    vb, logliks, test_logliks, r_vals = train_model(vb, n_iter, llk, test_data=test_data, no_base=no_base)

    title = f'Results - {llk} - R_max {max_r} - Clip value(F) {F_clamp}\n init lengthscale - {ell_init}'

    # Handle sparse and stochastic latents
    if isinstance(vb.latents, SparseLatents) or isinstance(vb.latents, SparseStochasticLatent):
        # expand the number of points from minibatch to all points
        vb.latents.update_sample_indices(list(range(vb.T)))
        logliks = []

    fig = plot_results(vb, logliks, r_vals, title)





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

    test_llks_ = None
    test_llk_mean, test_llk_std = None, None
    if test_data is not None:

        if llk == 'negbinom':
            tmp = nbinom.logpmf(test_data, detach(vb.dispersions.mean)[None, ..., None], 1 - prob)
        else:
            n = detach(vb.n)[None, ..., None]
            tmp = binom.logpmf(test_data, n, prob)

        test_llks_ = np.nan_to_num(tmp, neginf=0.)
        test_llk_mean, test_llk_std = test_llks_.mean(0).sum(), test_llks_.std(0).sum()

    print(f'Train llk {llk_mean} , Test llk {test_llk_mean}' )
    # end recalculating llk


    if output_dir:
        # save results
        fig.savefig(f'{output_dir}/summary.png')
        ell = detach(vb.latents.ell)

        plt.figure()
        psth(detach(vb.Y), detach(vb.firing_rate))
        plt.savefig(f'{output_dir}/firing_rate.png')


    firing_rates = detach(vb.firing_rate)
    firing_rates = zero_pad((len(non_zero_indices), vb.T),  firing_rates, non_zero_indices)
    assert firing_rates.shape == (len(non_zero_indices), vb.T)


    results = {
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
            'firing_rates': firing_rates,
            'elapsed_time': elapsed_time
        }




    return vb, results




if __name__ == '__main__':
    import argparse
    from pathlib import Path

    torch.manual_seed(0)

    parser = argparse.ArgumentParser(description='Synthetic Data experiments')
    parser.add_argument('--dataset', action='store', type=str,
                        default=None)
    parser.add_argument('--output_dir', action='store', type=str,
                        default=None)

    parser.add_argument('--test-size', action='store', type=int, default=0)
    parser.add_argument('--n-iter', action='store', type=int, default=50)  # latent dimensions
    parser.add_argument('--n-dim', action='store', type=int, default=10) # latent dimensions
    parser.add_argument('--ell', action='store', type=float, default=5)  # initial lengthscale values
    parser.add_argument('--r-max', action='store', type=float, default=10.) # max value for R
    parser.add_argument('--f-clamp', action='store', type=float, default=1.)
    parser.add_argument('--llk', action='store', type=str, default='negbinom')
    parser.add_argument('--no-base', action='store', type=bool, default=False)
    parser.add_argument('--single', action='store', type=bool, default=False)
    parser.add_argument('--sparse', action='store', type=bool, default=False)
    parser.add_argument('--lr', action='store', type=float, default=0.25)
    parser.add_argument('--batch', action='store', type=int, default=25)

    # number of inducing point every "n-inducing" points
    parser.add_argument('--n-inducing', action='store', type=int, default=None)


    args = parser.parse_args()

    # output directory
    output_dir = f'{args.output_dir}/{Path(args.dataset).stem}/{args.llk}___f_clamp{args.f_clamp}_n_inducing{args.n_inducing}'
    os.makedirs(output_dir, exist_ok=True)


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


    print('Using base ', args.no_base)

    results = {}

    vb, result = run(train_data,
             llk=args.llk,
             ell_init=args.ell,
             n_dim=args.n_dim,
             F_clamp=args.f_clamp,
             n_iter=args.n_iter, output_dir=output_dir, max_r=args.r_max, no_base=args.no_base, test_data=test_data,
             n_inducing=args.n_inducing, lr=args.lr, sparse=args.sparse, batch_size=args.batch)



    result['shuffle_seed'] = shuffle_seed


    # save all data
    with open(f'{output_dir}/summary.pkl', 'wb') as f_:
        pickle.dump(result, f_)

