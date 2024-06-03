

if __name__ == '__main__':

    import numpy as np
    import pickle
    import matplotlib.pyplot as plt
    import argparse
    import os



    parser = argparse.ArgumentParser(description='Synthetic experiments generator')
    parser.add_argument('--output-dir', action='store', type=str)
    parser.add_argument('--n-trials', action='store', type=int, default=10)
    parser.add_argument('--N', action='store', type=int, default=100)
    parser.add_argument('--T', action='store', type=int, default=300)
    args = parser.parse_args()

    # https://github.com/tachukao/mgplvm-pytorch/blob/bgpfa/examples_bGPFA/toy.py
    np.random.seed(1888)
    #### generate synthetic latents and tuning curves ####
    n_trials = args.n_trials
    n, T = args.N, args.T
    ts = np.arange(T)  # time points
    dts_2 = (ts[:, None] - ts[None, :]) ** 2  # compute dts for the kernel
    ell = [10, 10, 10]  # effective length scale
    d_true = len(ell)
    xs = []
    for e in ell:
        K = np.exp(-dts_2 / (2 * e ** 2))  # TxT covariance matrix
        L = np.linalg.cholesky(K + np.eye(T) * 1e-6)  # TxT cholesky factor
        xs.append((L @ np.random.normal(0, 1, (T))))  # DxT true latent states
    xs = np.vstack(xs)

    C = np.random.normal(0, 1, (n, d_true)) * 0.1  # factor matrix
    F = C @ xs  # n x T true de-noised activity

    # draw noise from NegBinomial model
    c_nb = -1.50  # scale factor for reasonable magnitude of activity
    p_nb = np.exp(F + c_nb) / (1 + np.exp(F + c_nb))  # probability of failure (c.f. Jensen & Kao et al.)
    r_nb = np.random.uniform(1, 10, n)  # number of failures (overdispersion paramer; c.f. Jensen & Kao et al.)
    # numpy defines it's negative binomial distribution in terms of #successes so we substitute 1 -> 1-p
    YNB = np.array([np.random.negative_binomial(r_nb, 1 - p_nb.T).astype(float).T for _ in range(n_trials)])

    plt.imshow(YNB[0], cmap='Greys', aspect='auto')
    plt.colorbar()
    plt.show()

    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    n_datasets = 5

    for i in range(n_datasets):
        time_steps = T - i * 300
        Y = YNB[:, :, :time_steps]

        print(time_steps, Y.shape  )

        with open(f'{output_dir}/data_{n_trials}_{n}_{time_steps}.pkl', 'wb') as f:
            pickle.dump({
                'data': Y.swapaxes(-1, -2),
                'W': C,
                'X': xs,
                'F': F[:, :time_steps],
                'r_nb': r_nb,
                'ell': ell
            }, f)
