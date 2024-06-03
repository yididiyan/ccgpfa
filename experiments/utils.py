from sklearn.linear_model import RidgeCV

import pickle
import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from ccgpfa.utils import hinton, detach
import torch
from tqdm import tqdm
from scipy.stats import nbinom, binom
from scipy.linalg import svd
import neo
import quantities as pq
plt.rcParams.update({'font.size': 22})


from ccgpfa.inference.stochastic_vb_binomial_multi_trials import StochasticSparseVariationalInference as StochasticNegBinom
from ccgpfa.inference.stochastic_vb_neg_binomial_multi_trials import StochasticSparseVariationalInference as StochasticBinomial



def load_kinematics_data(f, binsize=25, swapaxes=True):
    with open(f, 'rb') as f_:
        data = pickle.load(f_)

        Y = data['data'].astype(np.float32)
        if swapaxes: Y = Y.swapaxes(-1, -2)

    ts = np.arange(Y.shape[-1]) * binsize # measured in ms
    cs = CubicSpline(ts, data['locs'])  # fit cubic spline to behavior

    return Y, data['locs'], cs

def load_data(f, swapaxes=True):
    with open(f, 'rb') as f_:
        data = pickle.load(f_)

        Y = data['data'].astype(np.float32)
        if swapaxes: Y = Y.swapaxes(-1, -2)

    return Y
def decode(features, cs, binsize=25):
    """

    :param features: the spiking activity features
    :param cs: cubic spline we fit the location data with
    :param locs: locations
    :param targets: reach targets
    :return: R2 score
    """

    T = features.shape[0] # number of time steps
    ts = np.arange(T) * binsize

    alphas = list(np.logspace(-3, 3, 10))
    n_portions = 10

    n_elt_per_portion = T / n_portions
    all_indices = np.array([np.arange(n_portions) * n_elt_per_portion]).squeeze()
    train_indices = all_indices[0::2]
    test_indices = all_indices[1::2]

    train_indices = np.array([np.arange(t, t + n_elt_per_portion) for t in train_indices]).astype(int).flatten()
    test_indices = np.array([np.arange(t, t + n_elt_per_portion) for t in test_indices]).astype(int).flatten()


    delays = [100]  # add 100 ms to the data
    performance = np.zeros((len(delays), 2))  # model performance
    for idelay, delay in enumerate(delays):
        vels = cs(ts + delay, 1)  # velocity at time+delay
        for itest, Ytest in enumerate([features]):  # bGPFA
            features_train = features[train_indices]
            features_test = features[test_indices]
            vels_train = vels[train_indices]
            vels_test = vels[test_indices]

            regs = [RidgeCV(alphas=alphas, cv=10).fit(features_train, vels_train[:, idx]) for idx in
                    range(2)]  # fit x and y vel on half the data
            scores = [regs[idx].score(features_test, vels_test[:, idx]) for idx in
                      range(2)]  # score x and y vel on the other half
            performance[idelay, itest] = np.mean(scores)  # save performance

    max_delay, max_value = delays[np.argmax(performance[:, 0])], np.max(performance[:, 0])
    print(f'Max performance at {max_delay} -- max value {max_value}')

    return max_value



"""
Cosmoothing utilities 

"""
def split_data(Y, n_pieces=4):
    """
    Split observations into four set of neurons
    :param Y: M x N x T tensor
    :param n_pieces: number of chunks
    :return:
    """
    N = Y.shape[1]
    N_split = int(N / n_pieces)

    return [ Y[:, i: (i + N_split), :] for i in range(0, N, N_split)][: n_pieces]



def orthonormalize(C, X):

    U, s, Vh = svd(C, full_matrices=False)
    print(U.shape, s.shape, Vh.shape )
    # n, p = U.shape[0], s.shape[0]
    # S = np.zeros((n, p))
    # S[:p, :p] = s # prep big S

    X_orth = np.diag(s) @ Vh @ X

    # X_orth = Vh @ X
    C_orth = U

    return C_orth, X_orth


def plot_results(vb, logliks, r_vals, title):
    logliks = np.array(logliks)
    Y = detach(vb.Y)

    # orthonormalize
    W_orth, X_orth = orthonormalize(detach(vb.weights.mean), detach(vb.latents.mean))

    zero_index = []
    W_orth = detach(vb.weights.mean)

    # zero_index = list(np.abs(X_orth).sum(-1)).index(0.)
    # limit the number of neurons to the first 10
    # zero the columns of the weight matrix to eliminate the irrelevant latents
    # W_orth[:, zero_index:] = W_orth[:, zero_index:] * 0. + 1e-15

    # plot them figures
    fig, (ax1, ax2, ax3) = plt.subplots(3, 3, constrained_layout=True, figsize=(15, 15))
    hinton(W_orth[:10].T, ax=ax1[0])

    ax1[0].set_title('Weights (Unorthonormalized)')
    for n in range(vb.N):
        if np.any(logliks):
            ax1[1].plot(logliks[:, n])
        # if r_vals is not None:
        #     ax2[0].plot(r_vals[:, n])
    if np.any(logliks):
        ax1[2].plot(logliks.sum(-1))
        ax1[2].set_title(f'Total loglikelihood - {logliks.sum(-1)[-1]:.2f}')

    ax1[1].set_xlabel('iterations')
    ax1[1].set_title(f'llk (per neuron)')

    ax2[0].set_title('Dispersion values')


    prob = detach(vb.success_prob)

    if r_vals is not None:
        ax2[2].hist(Y[0].flatten(), label='true', alpha=0.5)
        # ax2[2].hist(nbinom.rvs(r_vals[-1][None, ..., None], 1 - prob).flatten(), label='inferred', alpha=0.5)
        # ax2[2].legend()
    else:
        pass
        # ax3[1].hist(Y[0].flatten(), label='true', alpha=0.5)
        # ax3[1].hist(nbinom.rvs(r_vals[-1][None, ..., None], 1 - prob).flatten(), label='inferred', alpha=0.5)
        # ax3[1].legend()

    ax3[0].plot(detach(vb.latents.mean).T)
    ax3[0].set_title('Latents')

    ax3[1].plot(X_orth.T)
    ax3[1].set_title('Latents (orthonormalized)')

    ax3[2].plot(X_orth[0], X_orth[1])
    ax3[2].set_title('Latents (orthonormalized) - 2D')



    fig.suptitle(title)
    return fig

def train_model_stochastic(vb, n_iter, llk, test_data=None, update_latents=True, update_weights=True, no_base=False, llk_every=50, threshold=0.1, update_r=10):

    logliks, test_logliks = [], []
    r_vals = []
    N = vb.N
    Y = detach(vb.Y)
    all_indices = list(range(0, vb.T))

    if no_base:
        vb.base.mean = 0. * vb.base.mean
        vb.base.covariance = 0 * vb.base.covariance

    for iter in tqdm(range(n_iter)):
        vb.update_sample_indices()
        if not no_base:
            vb.update_base()

        if update_latents:
            vb.update_latents()

        if update_weights:
            vb.update_weights()
        vb.update_augmented_vars()
        if llk == 'negbinom' and iter % update_r == 0: # update R
            vb.update_dispersion()


        if llk == 'negbinom':
            r_inferred = detach(vb.dispersions.mean)[None, ..., None]
            r_vals.append(r_inferred.squeeze())

        if iter % llk_every == 0:
            vb.sample_indices = all_indices
            vb.latents.update_sample_indices(all_indices)

            prob = detach(vb.success_prob)

            if llk == 'negbinom':
                logliks.append(nbinom.logpmf(Y, r_inferred, 1 - prob).sum(0).sum(-1))

            elif llk == 'binom':
                logliks.append(binom.logpmf(
                    Y[:, :, :],
                    detach(vb.n)[None, ..., None],
                    detach(vb.success_prob)[None, ...]).sum(axis=(0, 2)))

            if len(logliks) > 2:
                diff = (logliks[-1] - logliks[-2]).mean()
                print(diff)
                if np.abs(diff) < threshold:
                    print(diff)
                    break


    logliks = np.array(logliks)
    r_vals = np.array(r_vals)


    return vb, logliks, test_logliks, (None if llk == 'binom' else r_vals)



def train_model(vb, n_iter, llk, test_data=None, update_latents=True, update_weights=True, no_base=False, llk_every=50, threshold=0.1, update_r=10):
    if isinstance(vb, StochasticBinomial) or isinstance(vb, StochasticNegBinom):
        return train_model_stochastic(vb, n_iter, llk, test_data=test_data, update_latents=update_latents,
                                      update_weights=update_weights, no_base=no_base)

    logliks, test_logliks = [], []
    r_vals = []
    N = vb.N
    Y = detach(vb.Y)

    if no_base:
        vb.base.mean = 0. * vb.base.mean
        vb.base.covariance = 0 * vb.base.covariance

    for iter in tqdm(range(n_iter)):
        if not no_base:
            vb.update_base()

        if update_latents:
            vb.update_latents()
        if update_weights:
            vb.update_weights()
        vb.update_augmented_vars()
        if llk == 'negbinom' :

            vb.update_dispersion()




        if iter % llk_every == 0:
            prob = detach(vb.success_prob)


            if llk == 'negbinom':
                r_inferred = detach(vb.dispersions.mean)[None, ..., None]
                logliks.append(nbinom.logpmf(Y, r_inferred, 1 - prob).sum(0).sum(-1))
                r_vals.append(r_inferred.squeeze())

            elif llk == 'binom':
                logliks.append(binom.logpmf(
                    Y[:, :, :],
                    detach(vb.n)[None, ..., None],
                    detach(vb.success_prob)[None, ...]).sum(axis=(0, 2)))

            if len(logliks) > 2:
                diff = (logliks[-1] - logliks[-2]).mean()
                print(diff)
                if np.abs(diff) < threshold:
                    print(diff)
                    break

    logliks = np.array(logliks) / Y.shape[0]
    # test_logliks = np.array(test_logliks) / test_data.shape[0]

    r_vals = np.array(r_vals)


    return vb, logliks, test_logliks, (None if llk == 'binom' else r_vals)

#
# def train_model_stochastic(vb, n_iter, llk, test_data=None, update_latents=True, update_weights=True, no_base=False):
#
#     logliks, test_logliks = [], []
#     r_vals = []
#     N = vb.N
#     Y = detach(vb.Y)
#
#     if no_base:
#         vb.base.mean = 0. * vb.base.mean
#         vb.base.covariance = 0 * vb.base.covariance
#     # import ipdb; ipdb.set_trace()
#     for _ in tqdm(range(n_iter)):
#         vb.update_sample_indices()
#         if not no_base:
#             vb.update_base()
#
#         if update_latents:
#             vb.update_latents()
#         if update_weights:
#             vb.update_weights()
#         vb.update_augmented_vars()
#         if llk == 'negbinom':
#             vb.update_dispersion()
#
#         prob = detach(vb.success_prob)
#
#     #     if llk == 'negbinom':
#     #         r_inferred = detach(vb.dispersions.mean)[None, ..., None]
#     #         if test_data is not None:
#     #             test_logliks.append(nbinom.logpmf(test_data, r_inferred, 1 - prob).sum(0).sum(-1))
#     #         logliks.append(nbinom.logpmf(Y, r_inferred, 1 - prob).sum(0).sum(-1))
#     #         r_vals.append(r_inferred.squeeze())
#     #
#     #     elif llk == 'binom':
#     #         if test_data is not None:
#     #             test_logliks.append([binom.logpmf(
#     #             test_data[:, i, :],
#     #             detach(vb.n)[i],
#     #             detach(vb.success_prob[i])[None, ...]).sum() for i in range(N)])
#     #         logliks.append([binom.logpmf(
#     #             Y[:, i, :],
#     #             detach(vb.n)[i],
#     #             detach(vb.success_prob[i])[None, ...]).sum() for i in range(N)])
#     #
#     # logliks = np.nan_to_num(np.array(logliks), neginf=0.) / Y.shape[0]
#     # # test_logliks = np.nan_to_num(np.array(test_logliks), neginf=0.) / test_data.shape[0]
#     #
#     # r_vals = np.array(r_vals)
#
#
#     return vb, logliks, test_logliks, (None if llk == 'binom' else r_vals)
#



# def train_model(vb, n_iter, llk, test_data=None, update_latents=True, update_weights=True, no_base=False):
#     if isinstance(vb, StochasticBinomial) or isinstance(vb, StochasticNegBinom):
#         return train_model_stochastic(vb, n_iter, llk, test_data=test_data, update_latents=update_latents,
#                                       update_weights=update_weights, no_base=no_base)
#
#     logliks, test_logliks = [], []
#     r_vals = []
#     N = vb.N
#     Y = detach(vb.Y)
#
#     if no_base:
#         vb.base.mean = 0. * vb.base.mean
#         vb.base.covariance = 0 * vb.base.covariance
#
#     for _ in tqdm(range(n_iter)):
#         if isinstance(vb, StochasticBinomial) or isinstance(vb, StochasticNegBinom):
#             vb.update_sample_indices()
#         if not no_base:
#             vb.update_base()
#
#         if update_latents:
#             vb.update_latents()
#         if update_weights:
#             vb.update_weights()
#         vb.update_augmented_vars()
#         if llk == 'negbinom':
#             vb.update_dispersion()
#
#         prob = detach(vb.success_prob)
#
#         if llk == 'negbinom':
#             r_inferred = detach(vb.dispersions.mean)[None, ..., None]
#             # if test_data is not None:
#             #     test_logliks.append(nbinom.logpmf(test_data, r_inferred, 1 - prob).sum(0).sum(-1))
#             logliks.append(nbinom.logpmf(Y, r_inferred, 1 - prob).sum(0).sum(-1))
#             r_vals.append(r_inferred.squeeze())
#
#         elif llk == 'binom':
#             # if test_data is not None:
#             #     test_logliks.append([binom.logpmf(
#             #     test_data[:, i, :],
#             #     detach(vb.n)[i],
#             #     detach(vb.success_prob[i])[None, ...]).sum() for i in range(N)])
#             logliks.append([binom.logpmf(
#                 Y[:, i, :],
#                 detach(vb.n)[i],
#                 detach(vb.success_prob[i])[None, ...]).sum() for i in range(N)])
#
#     logliks = np.array(logliks) / Y.shape[0]
#     # test_logliks = np.array(test_logliks) / test_data.shape[0]
#
#     r_vals = np.array(r_vals)
#
#
#     return vb, logliks, test_logliks, (None if llk == 'binom' else r_vals)


"""
Taken from https://github.com/neurallatents/nlb_tools
"""
def array_to_spiketrains(array, bin_size):
    """Convert B x T x N spiking array to list of list of SpikeTrains"""
    stList = []
    for trial in range(len(array)):
        trialList = []
        for channel in range(array.shape[2]):
            times = np.nonzero(array[trial, :, channel])[0]
            counts = array[trial, times, channel].astype(int)
            times = np.repeat(times, counts)
            st = neo.SpikeTrain(times*bin_size*pq.ms, t_stop=array.shape[1]*bin_size*pq.ms)
            trialList.append(st)
        stList.append(trialList)
    return stList

def llk_and_time(files):
    times, llk = [], []
    for f in files:
        data= pickle.load(open(f, 'rb') )
        llk.append(data['train']['test_llk_mean'])
        times.append(data['train']['elapsed_time'])
    print(np.mean(times), np.std(times))
    print(np.mean(llk), np.std(llk))
