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
# def loss(A, x, y):
#     """
#     return 0.5 * ||Ax - y||_2^2
#     """
#     return 0.5 * LA.norm(A.dot(x) - y, 2) ** 2
#
#
# def hyperplane(y_proj_act, weights_act, gamma_k):
#     """
#     Do projection onto a hyperplane
#     parameter gamma_k: the radius of L1-ball
#     parameter y_proj_act: the point to be projected
#     """
#     scalar1 = np.dot(weights_act, y_proj_act) - gamma_k
#     scalar2 = sum(weights_act ** 2)
#     try:
#         scalar3 = np.divide(scalar1, scalar2)
#     except ZeroDivisionError:
#         print("Error! - derivation zero for scalar2 =", scalar2)
#         sys.exit(1)
#     x_sub = y_proj_act - scalar3 * weights_act
#     return x_sub, scalar3
#
#
# def WeightedL1Projection(y, weights, radius):
#     """
#     Do projection onto a weighted L1-ball
#     parameter radius: the radius of the L1-ball
#     parameter weights: the weight
#     parameter y: the point to be projected
#     """
#
#     y_sign = np.sign(y)
#     y_proj = y_sign * y
#     act_ind = range(len(y))
#     while True:
#         # calculate y_bar_act
#         y_proj_act = y_proj[act_ind]
#         weights_act = weights[act_ind]
#
#         x_sol_hyper, lamb = hyperplane(y_proj_act, weights_act, radius)
#         y_proj_act = np.maximum(x_sol_hyper, 0.0)
#         y_proj[act_ind] = y_proj_act
#
#         # update the active set
#         act_ind = y_proj > 0
#
#         inact_ind_cardinality = sum(x_sol_hyper < 0)
#         if inact_ind_cardinality == 0:
#             x_opt = y_proj * y_sign
#             break
#
#     return x_opt, lamb
#
#
# def Frank_Wolfe_Lp(A, x, y, mu, p, radius):
#     """
#     Solve the signal recovery problem as
#     min 0.5 * ||Ax - y||_2^2
#     s.t. ||x||_p^p <= r,
#     where 0 < p < 1.
#     parameter radius: the radius of the Lp-ball
#     parameter mu: the step-size of the projected gradient descent method
#     parameter y: the measurement of signal
#     parameter A: the measurement matrix
#     """
#
#     bisection_flag = 0      # the flag of whether bisection method succeeds
#     tol = 1e-8              # tolerance of numerical precision
#     iter = 0                # the number of total iteration
#     iter_proj = 0           # the number of the call of projected gradient descent method
#     iter_fw = 0             # the number of the call of Frank-Wolfe method
#     stopping_tol = 1e-5     # the tolerance of stopping criteria
#
#     history_res = []
#     history_obj = []
#
#     while True:
#         iter += 1
#         residual = A.dot(x) - y
#         grad = np.dot(A.T, A.dot(x) - y)    # compute the gradient of objective
#         # Check whether the current point is on the boundary of the Lp-ball
#         if abs(LA.norm(x, p) ** p - radius) <= tol:
#             """
#             If the current point is on the boundary of the Lp-ball, approximate the Lp-ball as
#             a weighted L1-ball and apply projected gradient method.
#             """
#             # iter_proj += 1
#             x_pre = x
#             z = x - mu * grad
#             act_ind = np.where(abs(x) > tol)[0]
#             for i in range(len(z)):
#                 if i not in act_ind:
#                     z[i] = 0
#             w = np.zeros(len(x))
#             w[act_ind] = p * abs(x[act_ind]) ** (p - 1)
#             radius_L1 = radius - LA.norm(x[act_ind], p) ** p + w[act_ind].dot(abs(x[act_ind]))
#
#             # Do Re-weighted L1 projection
#             x, lamb = WeightedL1Projection(z, w, radius_L1)
#
#             # Stopping criteria
#             proj_res = LA.norm(x - x_pre, 2)
#             ind_tmp = abs(x) > tol
#             global_res = LA.norm(A[:, ind_tmp].T.dot(A[:, ind_tmp].dot(x[ind_tmp]) - y) + lamb * np.sign(x[ind_tmp]) * abs(x[ind_tmp]) ** (p - 1), np.inf)
#             history_res.append(proj_res)
#             history_obj.append(loss(A, x, y))
#             print('{:5d}    {}:   Obj = {:3.3f}   Res = {:4.3e}   #nonzero = {}   global res = {}'.format(iter, 'Boundary case', loss(A, x, y), proj_res, len(np.nonzero(x)[0]), global_res))
#             if proj_res < stopping_tol:
#                 break
#
#         elif LA.norm(x, p) ** p < radius:
#             """
#             If the current point is inside the Lp-ball, apply Frank-Wolfe method.
#             """
#             # iter_fw += 1
#             grad_abs = abs(grad)
#             max_ind = np.argmax(grad_abs)
#
#             # Update the transformed variable
#
#             z[max_ind] = radius
#
#             # Update the original variable
#             s_abs = z ** (1 / p)
#             s = - s_abs * np.sign(grad)
#             d = s - x
#
#             fw_res = grad.dot(-d)
#             history_res.append(fw_res)
#             history_obj.append(loss(A, x, y))
#             print('{:5d}    {}:   Obj = {:3.3f}   Res = {:4.3e}   #nonzero = {}'.format(iter, 'Interior case', loss(A, x, y), fw_res, len(np.nonzero(x)[0])))
#             if fw_res < stopping_tol:
#                 break
#
#             # Determine whether find a gamma_bar such that the new iterate is on the boundary
#             gamma_amijo = fw_res / (LA.norm(A.dot(d), 2) ** 2)
#             # gamma_amijo = fw_res / (L * LA.norm(d) ** 2)
#             if LA.norm(x + gamma_amijo * d, p) ** p > radius:
#                 g_r = gamma_amijo
#                 g_l = 0.0
#                 gamma_bar = (g_l + g_r) / 2.0
#                 res = LA.norm(x + gamma_bar * d, p) ** p - radius
#                 while abs(res) > tol:
#                     if g_r - g_l <= 0:
#                         bisection_flag = 1
#                         break
#                     if res > 0:
#                         g_r = gamma_bar
#                     else:
#                         g_l = gamma_bar
#                     gamma_bar = (g_l + g_r) / 2.0
#                     res = LA.norm(x + gamma_bar * d, p) ** p - radius
#             else:
#                 gamma_bar = 1.0
#
#             if bisection_flag == 1:
#                 print('Can not find a root by bisection method!')
#                 print(LA.norm(x + gamma_bar * d, p) ** p - radius)
#                 break
#
#             # Get step-size for Frank-Wolfe method
#             gamma = min(1, gamma_bar, gamma_amijo)
#             x += gamma * d
#
#         else:
#             print('The new iterate is infeasible!')
#             break
#
#     return x, iter_fw, iter_proj, history_obj, history_res
#     # return x


if __name__ == '__main__':

    # Generate simulated data
    np.random.seed(1)
    random.seed(1)
    n = 32
    m = np.int32(np.linspace(54, 4 * (n - 4), 15))
    s = np.int32(np.linspace(2, n - 2, 15))  # number of zero components

    p = 0.5
    average_num = 10

    len_s = len(s)
    len_m = len(m)
    list_res = np.zeros((len_s, len_m, average_num))
    list_error = np.ones((len_s, len_m, average_num))

    for i in range(len_s):

        for j in range(len_m):

            for k in range(average_num):

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
