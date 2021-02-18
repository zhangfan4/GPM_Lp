import numpy as np
from numpy import linalg as LA
import GPM_Lp
import FW_Lp
import time
import data_loader
import random
import sys
from itertools import product
import pandas as pd


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


def loss(A, x, y):
    """
    return 0.5 * ||Ax - y||_2^2
    """
    return 0.5 * LA.norm(A.dot(x) - y, 2) ** 2


def main():
    """
        Solve the signal recovery problem as
        min 0.5 * ||Ax - y||_2^2
        s.t. ||x||_p^p <= r,
        where 0 < p < 1.
        :parameter Alg_flag: '0' for GPM, '1' for FW
    """

    '''Data Processing'''
    seeds = random.sample(range(1000), 10)   # 5 different random seeds
    # seeds = [151, 190, 250, 842, 935]
    row_list = []
    # seed = 15
    for seed in seeds:
        np.random.seed(seed)
        p = 0.5
        # List for saving evaluation metrics

        method_hyper = ['FW', 'GPM']
        # method_hyper = ['FW']
        m_hyper = [1000, 2000, 5000]  # # of measurements
        n_hyper = [1000, 2000, 5000]  # dimension of variable x
        param = {
            'method': 'XX',
            # 'm': 0,
            'n': 0
        }
        names = ['method', 'n']
        hypers = [method_hyper, n_hyper]

        '''pass through all combinations of initial points and methods'''
        for pack in product(*hypers):
            values = list(pack)
            print(values)
            for i in range(len(values)):
                param[names[i]] = values[i]

            # Generate simulated data
            n = param['n']
            m = n
            # s = int(0.2*n)
            # A, x_opt, y = data_loader.load_data(m, n, s)
            # radius = n - s
            A = np.random.randn(m, n)
            x_opt = np.random.rand(n)   # x_pot is ground truth
            y = A.dot(x_opt)
            radius = 0.8 * LA.norm(x_opt, p) ** p

            # Find the Lipschitz constant
            ATA = A.T.dot(A)
            bk = power_iteration(ATA, 100)  # Call power iteration alg.
            lambda_max = bk.T.dot(ATA).dot(bk) / bk.T.dot(bk)  # the largest eigenvalue value
            L = lambda_max
            alpha = 1 / L  # step-size of moving along the gradient
            # print(alpha, radius)

            '''Run GPM or FW'''
            x = np.zeros(n)

            if param['method'] is 'GPM':
                start = time.time()
                x, iter_gpm, iter_proj = GPM_Lp.GPM(A, x, y, alpha, p, radius)
                total_iter = iter_proj + iter_gpm
                print('-' * 40)
                # print('Total time GPM:', time.time()-start)
                print('#GPM = {:5d}   #proj = {:5d}'.format(iter_gpm, iter_proj))

            elif param['method'] is 'FW':
                start = time.time()
                x, iter_fw, iter_proj = FW_Lp.Frank_Wolfe_Lp(A, x, y, alpha, p, radius)  # FW
                total_iter = iter_proj + iter_fw
                print('-' * 40)
                # print('Total time FW:', time.time() - start)
                print('#fw = {:5d}   #proj = {:5d}'.format(iter_fw, iter_proj))
            else:
                start = time.time()
                total_iter = 0
                print('This method is not provided')

            num_nonzero = len(np.nonzero(x)[0])
            obj_val = loss(A, x, y)
            total_time = time.time() - start

            print('Saving results ......')
            save_data = {}
            save_data['seed'] = seed
            save_data['method'] = param['method']
            save_data['m'] = m
            save_data['n'] = n
            save_data['# iter'] = total_iter
            save_data['# nonzero'] = num_nonzero
            save_data['obj'] = obj_val
            save_data['CPU time'] = total_time
            row_list.append(save_data)

    # Saving results to a csv file
    dfsave_data = pd.DataFrame(data=row_list)
    filename = 'p: %3.2f' % p + '_result.csv'
    # filename = 'result.csv'
    dfsave_data.to_csv(filename, index=True)

    print('DONE')
    return 0


main()
