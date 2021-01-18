# -*- coding: utf-8 -*-
# @Time    : 2021/1/6 19:38
# @Author  : Xin Deng
# @FileName: PGM_FW.py


import numpy as np
import sys, time, random
import matplotlib.pyplot as plt
from numpy import linalg as LA
import scipy.sparse
from scipy.stats import rv_continuous
import data_loader
import FW_Lp
import GPM_L1
import GPM_Lp


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


if __name__ == '__main__':

    # Generate simulated data
    np.random.seed(1)
    random.seed(1)
    n = 32
    point = 8
    m = np.int32(np.linspace(int(n/point), 4 * n, point))
    s = np.int32(np.linspace(2, n-2, point))  # number of zero components

    p = 0.5
    average_num = 5

    len_s = len(s) 
    len_m = len(m)
    list_res = np.zeros((len_s, len_m, average_num))
    list_error = np.ones((len_s, len_m, average_num))

    for i in range(len_s):

        for j in range(len_m):

            for k in range(average_num):
                # data generation

                # x_opt = np.random.randn(n)
                # # print(x_opt)
                # zero_index = random.sample(range(n), s[i])
                # x_opt[zero_index] = 0
                A, x_opt, y = data_loader.load_data(m[j], n, s[i])
                # A = np.identity(m[j])
                # y = A.dot(x_opt)
                # y = np.array([0.5, 0.45])
                # radius = 0.5 * LA.norm(x_opt, p) ** p
                # radius = LA.norm(x_opt, 1) ** p * n ** (1 - p)
                radius = (n - s[i])

                # Find the Lipschitz constant
                ATA = A.T.dot(A)
                bk = power_iteration(ATA, 100)  # Call power iteration alg.
                lambda_max = bk.T.dot(ATA).dot(bk) / bk.T.dot(bk)  # the largest eigenvalue value
                L = lambda_max
                mu = 1. / L  # step-size of outer algorithm
                # print(mu, radius)

                # Initialization
                x = np.zeros(n)
                # x = np.random.rand(n)
                # x = (radius / LA.norm(x, p) ** p) ** (1/p) * x
                # x = np.ones(n) * (radius / n) ** (1 / p)
                # x = (radius / LA.norm(A.T.dot(y), p) ** p) ** (1/p) * A.T.dot(y)
                # print(radius, LA.norm(x, p) **p)

                t_start = time.time()
                # x, iter_fw, iter_proj, history_obj, history_res = FW_Lp.Frank_Wolfe_Lp(A, x, y, mu, p, radius)  # Lp FW
                # x = GPM_L1.GPM(A, x, y, mu, radius)  # L1 GPM
                x = GPM_Lp.GPM(A, x, y, mu, p, radius)  # Lp GPM

                t_end = time.time()
                # print(t_end - t_start)

                # Check the location of nonzero elements
                # print(x_opt)
                # print(x)
                # print(np.nonzero(x_opt)[0])
                # print(np.nonzero(x)[0])
                # print(len(np.nonzero(x_opt)[0]))
                # print(len(np.where(abs(x) > 1e-4)[0]))
                # print(x)
                # print('---------')
                # print(x_opt)
                
                set_x_opt = set(np.nonzero(x_opt)[0])
                set_x = set(np.nonzero(x)[0])
                if set_x & set_x_opt == set_x_opt:
                    print('Check the location of nonzero elements: Success')
                else:
                    print('Check the location of nonzero elements: Failed')

                list_error[i, j, k] = LA.norm(x - x_opt) / LA.norm(x_opt)
                if list_error[i, j, k] < 1e-3:
                    list_res[i, j, k] = 1

                print('-' * 40)
                # print('Ground Truth:', x_opt)
                # print('Solution:', x)
                # print('||x - x_opt||_inf:', LA.norm(x - x_opt, np.inf))
                # print('Relative error:', error)
                print('s = {:3d}    m = {:4d}    k = {:2d}   Relative error = {:3.3e}   result = {:2f}'.format(s[i], m[j], k, list_error[i, j, k], list_res[i, j, k]))

    np.save('results/list_error_100.npy', list_error)
    np.save('results/list_res_100.npy', list_res)

    fig = fig = plt.figure(1)
    curve1 = plt.plot(range(len(history_res)), history_res, label = r'$\alpha(x^k)$')
    curve2 = plt.plot(range(len(history_obj)), history_obj, label = r'$\alpha(x^k)$')
    # plt.yscale('log')
    plt.xlabel('Number of iterations')
    plt.ylabel('Relative FW gap')
    plt.title('FW on a Lp-ball constrained problem')
    # plt.xlim((0, 100))
    first_legend = plt.legend(handles=[curve1, curve2], loc='upper right', shadow=True)
    plt.grid()
    plt.show()
