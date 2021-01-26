import numpy as np
#import matplotlib.pyplot as plt
from numpy import linalg as LA
# from sklearn import preprocessing
import time
from data_loader import load_data
import scipy.linalg


def loss(x, A, b):
    """Least Square loss function
    :param x: parameter vector of dimension n by 1
    :param A: feature matrix dimension m by n where m is the number of data
              points and n is the number of features.
    :param b: label vector of dimension m by 1.
    :return: loss
    """

    return 0.5 * LA.norm(b-A.dot(x)) ** 2

# print loss(theta, x, y)


def gradient(x, A, b):
    """Gradient of logistic regression loss.
    :param x: parameter vector of dimension n by 1
    :param A: feature matrix dimension m by n where m is the number of data
              points and n is the number of features.
    :param b: label vector of dimension m by 1.
    :return: gradient of Least Square loss
    """
    return -A.T.dot(b-A.dot(x))


def hyperplane(x, tau):
    """
    Projection onto the surface of l1 ball .
    :param x: being projected point
    :param tau: radius of L1 ball
    :return: the projection z
    """
    dim = np.size(x, 0)
    w = x - 1.0 * (sum(x) - tau)/dim * np.ones(dim)
    # print w
    return w


def project(tau, x, A, y, beta):
    """L1 ball exact projection
    :param tau: L1 ball radius.
    :param x: parameter vector of dimension n by 1 to be projected
    :param g: gradient g to be projected
    :param beta: stepsize of outer algorithm 
    :return: projection of theta onto l1 ball ||x||_1 <= tau, and the iteration k
    """

    g = gradient(x, A, y)
    v = x - beta * g  # being projected point
    # print LA.norm(v, 1)
    n = np.size(v, 0)

    # exact projection
    if LA.norm(v, 1) <= tau:
        k = 0
        z = v
        # print('It is already in the L1 ball.')
        return z
    else:
        # print('It is not in the L1 ball.')
        signum = np.sign(v)  # record the signum of v
        max_iter = 1e3
        v_temp = signum * v  # project v onto R+, named v_temp
        act_ind = range(n)  # record active index

        for i in range(int(max_iter)):

            # calculate v_temp_act
            # v_temp_act = np.zeros((len(act_ind), 1))
            v_temp_act = v_temp[act_ind]

            w = hyperplane(v_temp_act, tau)  # projection onto hyperplane

            # update v_temp_act
            v_temp_act = np.maximum(w, 0)
            # update v_temp
            v_temp[act_ind] = v_temp_act

            # update act_ind
            act_ind = []
            for ind in range(n):
                if v_temp[ind] > 0:
                    act_ind.append(ind)
            # print act_ind

            # termination criterion
            if sum(w<0) == 0:
                k = i + 1
                z = signum * v_temp
                return z


# def IGPM(tau, x, g, beta, gamma, iter, omega):
#     """L1 ball inexact projection
#     :param tau: L1 ball radius.
#     :param x: parameter vector of dimension n by 1 to be projected
#     :param g: gradient g to be projected
#     :param beta: stepsize of outer algorithm
#     :return: projection of theta onto l1 ball ||x||_1 <= tau, and the iteration k
#     """
#     v = x - beta * g  # being projected point
#     n = np.size(v, 0)
#     omega /= ((iter + 1) ** 2)
#     # print n
#
#     # exact projection
#     if LA.norm(v, 1) <= tau:
#         k = 0
#         z = v
#         # print('It is already in the L1 ball.')
#         return z, k
#     else:
#         # print('It is not in the L1 ball.')
#         signum = np.sign(v)  # record the signum of v
#         max_iter = 1e3
#
#         v_temp = signum * v  # project v onto R+, named v_temp
#         act_ind = range(n)  # record active index
#
#         # calculate the initial value of primal dual objectives
#         p0 = 0.5 * beta**2 * LA.norm(g)**2
#         q0 = 0.5 * LA.norm(v)**2
#
#         for i in range(int(max_iter)):
#
#             # calculate v_temp_act
#             # v_temp_act = np.zeros((len(act_ind), 1))
#             v_temp_act = v_temp[act_ind]
#
#             w = hyperplane(v_temp_act, tau)  # projection onto hyperplane
#             # print('w1:%d, w2:%d' % (w[0], w[1]))
#
#             # update v_temp_act
#             v_temp_act = np.maximum(w, 0)
#             # update v_temp
#             v_temp[act_ind] = v_temp_act
#
#             # update act_ind
#             act_ind = []
#             for ind in range(n):
#                 if v_temp[ind][0] > 0:
#                     act_ind.append(ind)
#             # print act_ind
#
#             y = tau / LA.norm(v_temp, 1) * v_temp  # scaling into L1 ball
#             z_hat = signum * y  # project back onto R
#
#             # calculate the primal-dual ratio
#             zz = 0.5 * LA.norm(z_hat)**2  # 0.5||z||^2
#             p = zz - np.dot(z_hat.T, v) + q0
#             q = -zz - tau * LA.norm(v-z_hat, np.inf) + q0
#             ratio = (p0 - p + omega) / (p0 - q + omega)
#             # print('delta_p:%f, delta_q:%f, ratio:%f' % (p0 - p, p0 - q, ratio))
#
#             # termination criterion
#             if ratio >= gamma:
#                 k = i + 1
#                 z = z_hat
#                 return z, k
def GPM(A, x, b, beta, tau):
    """
        gradient projection algorithm for least square problem with L1 ball constraint
    """

    max_iter = int(1e4)
    eps = 1e-4  # tolerance of LS

    # outer algorithm
    count = 0

    while True:

        # choose GPM or IGPM
        count += 1
        z = project(tau, x, A, b, beta)

        # print LA.norm(z, 1)
        d = z - x  # searhcing direction d
        norm_d = LA.norm(d)

        # termination criterion
        if norm_d <= eps or count >= max_iter:
            if count >= max_iter:
                print('Not optimal!')
            x_hat = x  # final solution
            break

        x = z  # Update parameter theta

    return x_hat

