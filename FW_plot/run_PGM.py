"""
Run Projection Gradient Method (PGM) to solve the minimization problem on the L_p norm ball with radius r
formally, 
    min     f(x)
    s.t.    ||x||_p^p <= r 
"""


import numpy as np
import sys, time, random
import matplotlib.pyplot as plt
from numpy import linalg as LA
import scipy.sparse
from scipy.stats import rv_continuous
import data_loader
# import FW_Lp
# import GPM_L1
import GPM_Lp
from eig_computation import compute_largest_eigenvalue


if __name__ == '__main__':
    # Generate simulated data
    np.random.seed(1)
    random.seed(1)
    # m, n is the size of the measurement matrix A
    n = 32
    m = np.int32(np.linspace(54, 4 * (n - 4), 15))
    s = np.int32(np.linspace(2, n - 2, 15))  # number of zero components

    p = 0.5 # value of p-norm
    average_num = 10

    len_s = len(s) 
    len_m = len(m)
    list_res = np.zeros((len_s, len_m, average_num))

    # the error is defined as the relative error
    # error =  norm(x - x_opt) / norm(x)
    list_error = np.ones((len_s, len_m, average_num))

    for i in range(len_s):

        for j in range(len_m):

            for k in range(average_num):
                # data generation
                A, x_opt, y = data_loader.load_data(m[j], n, s[i])
                radius = (n - s[i])

                # Find the Lipschitz constant, which is given by the largest eigenvalue of A^TA
                lambda_max = compute_largest_eigenvalue(A.T.dot(A), 100)
                L = lambda_max

                # step-size of algorithm
                step_size = 1. / L  

                # Initialization
                x0 = np.zeros(n)

                t_start = time.time()

                # Use GPM to solve the problem
                x = GPM_Lp.GPM(A, x0, y, step_size, p, radius)  

                t_end = time.time()

                # time spend
                time_used = t_end - t_start
                

                # check if the solution generated by our algorithm and the optimal solution have 
                # the same non-zero elements
                set_x_opt = set(np.nonzero(x_opt)[0])
                set_x = set(np.nonzero(x)[0])

                if set_x == set_x_opt:
                    print('Check the location of nonzero elements: Success')
                else:
                    print('Check the location of nonzero elements: Failed')

                list_error[i, j, k] = LA.norm(x - x_opt) / LA.norm(x_opt)
                if list_error[i, j, k] < 1e-3:
                    list_res[i, j, k] = 1
                
                # print the info of current iteration
                print('-' * 40)
                print('s = {:3d}    m = {:4d}    k = {:2d}   Relative error = {:3.3e}   result = {:2f}'.format(s[i], m[j], 
                        k, list_error[i, j, k], list_res[i, j, k]))



    # # %% plot the results
    # np.save('results/list_error_100.npy', list_error)
    # np.save('results/list_res_100.npy', list_res)

    # fig = fig = plt.figure(1)
    # curve1 = plt.plot(range(len(history_res)), history_res, label = r'$\alpha(x^k)$')
    # curve2 = plt.plot(range(len(history_obj)), history_obj, label = r'$\alpha(x^k)$')
    # # plt.yscale('log')
    # plt.xlabel('Number of iterations')
    # plt.ylabel('Relative FW gap')
    # plt.title('FW on a Lp-ball constrained problem')
    # # plt.xlim((0, 100))
    # first_legend = plt.legend(handles=[curve1, curve2], loc='upper right', shadow=True)
    # plt.grid()
    # plt.show()
