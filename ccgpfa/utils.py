import torch
from typing import Optional, List

from torch import Tensor
import numpy as np
from scipy.linalg import svd
import matplotlib.pyplot as plt

default_jitter = 1E-8


def softplus(x):
    return torch.log(1 + torch.exp(x))


def inv_softplus(x):
    return torch.log(torch.exp(x) - 1)

def safe_softplus(x):
    return torch.logaddexp(torch.zeros(x.shape).to(x.device), x)

def safe_inv_softplus(x):
    return torch.log(torch.expm1(x))

def get_device(device: str = "cuda"):
    if torch.cuda.is_available() and device == "cuda":
        my_device = torch.device(device)
    else:
        my_device = torch.device('cpu')
        # need to allow multiple instances of openMP
        import os
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    return my_device


# helper functions
def detach(tensor):
    """turns a pytorch parameter tensor into a numpy array (useful for plotting)"""
    return tensor.detach().cpu().numpy()


def split_data(data, N):
    prev_index = 0
    data_ = []
    for n in N:
        data_.append(data[:, prev_index: prev_index + n, :])
        prev_index = prev_index + n

    return data_


def diag_stack_tensors(tensors: List[Tensor], dim_1=-1, dim_2=-2, device=None):
    """
    Stack tensors diagonally, filling off-diagonals with zeros
    Args:
        tensors:
        dim_1: horizontal stacking done along this dimension index
        dim_2: vertical stacking done along this dimension index

    Returns: a block diagonal matrix with the given tensors.

    """
    device = device or get_device()

    result = []

    for i, t_i in enumerate(tensors):
        tensor_i = []  # tensor at layer i

        for j, t_j in enumerate(tensors):

            if i == j:
                tensor_i.append(t_j)
            else:
                shape = list(t_i.shape)
                shape[dim_1] = t_j.shape[dim_1]
                tensor_i.append(torch.zeros(shape, device=device))

        tensor_i = torch.cat(tensor_i, dim_1)
        result.append(tensor_i)

    result = torch.cat(result, dim_2)  # check

    return result


# GP covariance utilities
def construct_cov_matrix(T, sig=1., ell=1., shift=0):
    t = np.arange(0, T, 1).reshape(-1, 1).repeat(T, 1)
    dt = t - t.T
    K = sig ** 2 * np.exp(-(dt + shift) ** 2 / (2 * ell ** 2))

    return K


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




def hinton(matrix, max_weight=None, ax=None):
    """Draw Hinton diagram for visualizing a weight matrix."""
    ax = ax if ax is not None else plt.gca()

    if not max_weight:
        max_weight = 2 ** np.ceil(np.log2(np.abs(matrix).max()))

#     ax.patch.set_facecolor('gray')
    ax.set_aspect('equal', 'box')
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())

    for (x, y), w in np.ndenumerate(matrix):
        color = 'blue' if w > 0 else 'red'
        size = np.sqrt(abs(w) / max_weight)
        rect = plt.Rectangle([x - size / 2, y - size / 2], size, size,
                             facecolor=color, edgecolor=color)
        ax.add_patch(rect)

    ax.autoscale_view()
    ax.invert_yaxis()

