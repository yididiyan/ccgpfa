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

from run_with_behavior_mapping import zero_pad



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
    parser.add_argument('--bin-size', action='store', type=int, default=15)
    # best latent dimensions(choosen based on ARD methods)
    parser.add_argument('--n-dim', action='store', type=int, default=6)
    # threshold to remove neurons entry if has number of spikes below the "threshold"
    parser.add_argument('--threshold', action='store', type=int, default=0)

    args = parser.parse_args()

    bin_size_ms = args.bin_size

    # output directory
    output_dir = f'{args.output_dir}/{Path(args.dataset).stem}/gaussian'
    os.makedirs(output_dir, exist_ok=True)

    Y = load_data(args.dataset)
    Y = Y.swapaxes(-1, -2) # the spike train converter expects batch x timesteps x channels(neurons)

    # shuffle trials
    shuffle_seed = 0
    rng = np.random.default_rng(shuffle_seed)
    rng.shuffle(Y)
    if args.test_size:
        train_data = Y[:-args.test_size, ...]
        test_data = Y[-args.test_size:, ...]
        print(f'train size {train_data.shape[0]}, test size {test_data.shape[0]}')
    else:
        train_data = Y
        test_data = None


    np.random.seed(0) # seed for consistency
    train_data[..., train_data.sum(axis=(0, 1)) <= args.threshold] = 0.
    train_st = array_to_spiketrains(train_data, bin_size_ms)
    gpfa = GPFA(bin_size=(bin_size_ms * pq.ms), x_dim=args.n_dim, verbose=True)

    start_time = time.time()
    train_factors = gpfa.fit_transform(train_st)

    train_mean_trajectory = np.mean(train_factors, axis=0) # average trajectory along the trials

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
    non_zero_indices = Y.sum(-2).sum(0) > 0
    rates = np.array([zero_pad((len(non_zero_indices), T), rates[i], non_zero_indices) for i in range(len(rates))])
    assert rates[:, ~non_zero_indices, :].sum() == 0.
    assert rates.shape == (1, N, T)

    elapsed_time = time.time() - start_time
    test_llk, test_llk_mean, test_llk_std = None, None, None

    if test_data is not None:
        test_llk = -poisson.logpmf(test_data.swapaxes(-1, -2), rates)
        test_llk_mean, test_llk_std = test_llk.mean(), test_llk.std() / np.sqrt(test_llk.reshape(-1).shape[-1])

    train_llk = -poisson.logpmf(train_data.swapaxes(-1, -2), rates)
    train_llk_mean, train_llk_std = train_llk.mean(), train_llk.std() / np.sqrt(train_llk.reshape(-1).shape[-1])

    print(test_llk_mean, 'Sum of test llk --- time taken ', elapsed_time)
    print('Train Vs Test NLL')
    print(train_llk_mean, train_llk_std, test_llk_mean, test_llk_std)

    # import matplotlib.pyplot as plt
    # plt.boxplot(test_llk.reshape(-1))
    # plt.title('NLL - Gaussian')
    # plt.savefig(f'{output_dir}/test_llk.png')


    with open(f'{output_dir}/summary.pkl', 'wb') as f_:
        result = {
            'rates': rates,
            'test_llk_mean': test_llk_mean,
            'test_llk_std': test_llk_std,
            'test_llk': test_llk,
            'elapsed_time': elapsed_time
        }
        pickle.dump(result, f_)

