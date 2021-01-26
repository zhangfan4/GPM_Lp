import numpy as np
from numpy import linalg as LA
import GPM_Lp
import FW_Lp
import time
import sys


def power_iteration(A, num_simulations):
    """
    Power iteration Algorithm for largest eigenvalue
    :param A: a diagonalizable matrix
    :param num_simulations: max iteration number
    :return: eigenvector b_k
    """
    b_k = np.random.rand(A.shape[1])

    for _ in range(num_simulations):
        # calculate the matrix-by-vector product Ab
        b_k1 = A.dot(b_k)

        # calculate the norm
        b_k1_norm = LA.norm(b_k1)

        # re normalize the vector
        b_k = b_k1 / b_k1_norm

    return b_k


def main(Alg_flag):
    """
        Solve the signal recovery problem as
        min 0.5 * ||Ax - y||_2^2
        s.t. ||x||_p^p <= r,
        where 0 < p < 1.
        :parameter Alg_flag: '0' for GPM, '1' for FW
    """

    '''Data Processing'''
    # Generate simulated data
    np.random.seed(1)
    m, n = 1000, 1000
    p = 0.5

    A = np.random.randn(m, n)
    x_opt = np.random.rand(n)   # x_pot is ground truth
    # x_opt = np.array([0.,  1.,  0.,  -1.,  0.])
    # epsilon = np.random.randn(n) * 1e-5
    y = A.dot(x_opt)
    radius = 0.8 * LA.norm(x_opt, p) ** p

    # Find the Lipschitz constant
    ATA = A.T.dot(A)
    bk = power_iteration(ATA, 100)  # Call power iteration alg.
    lambda_max = bk.T.dot(ATA).dot(bk) / bk.T.dot(bk)  # the largest eigenvalue value
    L = lambda_max
    alpha = 1 / L  # step-size of moving along the gradient
    print(alpha, radius)

    '''Run GPM or FW'''
    x = np.zeros(n)

    if Alg_flag == 0:
        start = time.time()
        x, iter_gpm, iter_proj = GPM_Lp.GPM(A, x, y, alpha, p, radius)
        print('-' * 40)
        print('Total time GPM:', time.time()-start)
        print('#GPM = {:5d}   #proj = {:5d}'.format(iter_gpm, iter_proj))

    else:
        start = time.time()
        x, iter_fw, iter_proj = FW_Lp.Frank_Wolfe_Lp(A, x, y, alpha, p, radius)  # FW
        print('-' * 40)
        print('Total time FW:', time.time() - start)
        print('#fw = {:5d}   #proj = {:5d}'.format(iter_fw, iter_proj))

    # print('Ground Truth:', x_opt)
    # print('Solution:', x)


# main(1)  # for testing FW
main(0)  # for testing GPM

