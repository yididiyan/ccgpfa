# ---- Imports ---- #
import numpy as np
import pandas as pd
import h5py
import neo
import quantities as pq
from elephant.gpfa import GPFA
from sklearn.linear_model import LinearRegression, PoissonRegressor, Ridge

from nlb_tools.nwb_interface import NWBDataset
from nlb_tools.make_tensors import make_train_input_tensors, make_eval_input_tensors, make_eval_target_tensors, save_to_h5
from nlb_tools.evaluation import evaluate

from experiments.utils import load_data, array_to_spiketrains





if __name__ == '__main__':
    import argparse
    import pickle
    import os
    import time
    from pathlib import Path
    from scipy.stats import poisson


    parser = argparse.ArgumentParser(description='Allen Data experiments')
    parser.add_argument('--dataset', action='store', type=str,
                        default=None)
    parser.add_argument('--output_dir', action='store', type=str,
                        default=None)
    parser.add_argument('--test-size', action='store', type=int, default=25)
    # best latent dimensions(choosen based on ARD methods)
    parser.add_argument('--n-dim', action='store', type=int, default=6)

    bin_size_ms = 15

    args = parser.parse_args()

    # output directory
    output_dir = f'{args.output_dir}/{Path(args.dataset).stem}/gaussian'
    os.makedirs(output_dir, exist_ok=True)

    Y = load_data(args.dataset)
    Y = Y.swapaxes(-1, -2) # the spike train converter expects batch x timesteps x channels(neurons)

    # shuffle trials
    shuffle_seed = 0
    rng = np.random.default_rng(shuffle_seed)
    rng.shuffle(Y)

    train_data = Y[:-args.test_size, ...]
    test_data = Y[-args.test_size:, ...]
    print(f'train size {train_data.shape[0]}, test size {test_data.shape[0]}')


    np.random.seed(0) # seed for consistency

    train_st = array_to_spiketrains(train_data, bin_size_ms)
    test_st = array_to_spiketrains(test_data, bin_size_ms)
    gpfa = GPFA(bin_size=(bin_size_ms * pq.ms), x_dim=args.n_dim, verbose=True)

    start_time = time.time()
    train_factors = gpfa.fit_transform(train_st)

    train_mean_trajectory = np.mean(train_factors, axis=0)

    Y_estimated = (gpfa.params_estimated['C'] @ train_mean_trajectory) + gpfa.params_estimated['d'][..., None]
    noise = np.diag(gpfa.params_estimated['R'])
    N = Y.shape[2]
    T = Y.shape[1]
    print('Sampling firing rates')
    rates = []
    assert len(noise) == len(Y_estimated)
    n_samples = 20
    for n in range(Y_estimated.shape[0]):
        # for each neuron
        samples_n = np.array([np.random.multivariate_normal(Y_estimated[n], noise[n] * np.eye(T)) for i in range(n_samples)])
        assert samples_n.shape == (n_samples, T)
        samples_n[samples_n < 0.] = 1e-8 # turn negative spikes to small positive number
        samples_n = np.square(samples_n) # spike observations
        rates.append(np.mean(samples_n, 0))

    rates = np.array(rates)[None, ...]
    assert rates.shape == (1, N, T)

    elapsed_time = time.time() - start_time

    test_llk = -poisson.logpmf(test_data.swapaxes(-1, -2), rates)
    train_llk = -poisson.logpmf(train_data.swapaxes(-1, -2), rates)

    # test_llk_mean, test_llk_std = test_llk.mean(0).sum(), test_llk.sum(axis=(1,2)).std()
    # train_llk_mean, train_llk_std = train_llk.mean(0).sum(), train_llk.sum(axis=(1, 2)).std()

    test_llk_mean, test_llk_std = test_llk.mean(), test_llk.std() / np.sqrt(test_llk.reshape(-1).shape[-1])
    train_llk_mean, train_llk_std = train_llk.mean(), train_llk.std() / np.sqrt(train_llk.reshape(-1).shape[-1])

    print(test_llk_mean, 'Sum of test llk --- time taken ', elapsed_time)
    print('Train Vs Test NLL')
    print(train_llk_mean, train_llk_std, test_llk_mean, test_llk_std)

    import matplotlib.pyplot as plt
    plt.boxplot(test_llk.reshape(-1))
    plt.title('NLL - Gaussian')
    plt.savefig(f'{output_dir}/test_llk.png')


    with open(f'{output_dir}/summary.pkl', 'wb') as f_:
        result = {
            'rates': rates,
            'test_llk_mean': test_llk_mean,
            'test_llk_std': test_llk_std,
            'test_llk': test_llk,
            'elapsed_time': elapsed_time
        }
        pickle.dump(result, f_)

