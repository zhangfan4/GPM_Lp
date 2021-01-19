"""
This file is used to generate the data for experiments.
"""


import numpy as np
import random
import scipy.sparse
# from scipy.sparse import random
from scipy.stats import rv_continuous
import scipy.linalg


class CustomDistribution(rv_continuous):
    def _rvs(self, *args, **kwargs):
        return self._random_state.randn(*self._size)


def load_data(m, n, s):
    """
    :param m: number of samples
    :param n: number of features
    :param s: number of zero components
    :return: true value x0 with s zero components, and nonzero components are 1 or -1 with 50% probability, 
             sparse parameter matrix A,
             measurement vector b = A * x0
    """
    x0 = np.zeros(n)
    nonzero_index = random.sample(range(n), n-s)
    positive_index = random.sample(nonzero_index, int((n-s)/2))
    for i in nonzero_index:
        if i in positive_index:
            x0[i] = 1
        else:
            x0[i] = -1

    # dense random matrix
    A = np.random.normal(0, 1, size=(m, n))

    # A = scipy.linalg.orth(A)

    # sparse matrix
    # row = random.sample(range(min(m, n)), int(0.5*min(m, n)))  # row index
    # col = random.sample(range(min(m, n)), int(0.5*min(m, n)))  # column index
    # value = np.random.normal(0, 1, size=int(0.5*min(m, n)))  # data value
    # A = csr_matrix((value, (row, col)), shape=(m, n))
    # A = scipy.sparse.random(m, n, density=1.0/min(m, n), format="csr")

    # p = CustomDistribution()
    # q = p()  # get a frozen version of the distribution
    # A = scipy.sparse.random(m, n, density=1e-4, data_rvs=q.rvs)

    # # v = np.random.normal(0, 1, size=(m, 1))
    b = A.dot(x0)
    return A, x0, b
