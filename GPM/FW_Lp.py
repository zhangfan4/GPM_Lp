# -*- coding: utf-8 -*-
# @Time    : 2021/1/6 19:38
# @Author  : Xin Deng
# @FileName: FW_Lp.py


import numpy as np
import sys
from numpy import linalg as LA


def loss(A, x, y):
    """
    return 0.5 * ||Ax - y||_2^2
    """
    return 0.5 * LA.norm(A.dot(x) - y, 2) ** 2


def hyperplane(y_proj_act, weights_act, gamma_k):
    """
    Do projection onto a hyperplane
    parameter gamma_k: the radius of L1-ball
    parameter y_proj_act: the point to be projected
    """
    scalar1 = np.dot(weights_act, y_proj_act) - gamma_k
    scalar2 = sum(weights_act ** 2)
    try:
        scalar3 = np.divide(scalar1, scalar2)
    except ZeroDivisionError:
        print("Error! - derivation zero for scalar2 =", scalar2)
        sys.exit(1)
    x_sub = y_proj_act - scalar3 * weights_act
    return x_sub, scalar3


def WeightedL1Projection(y, weights, radius):
    """
    Do projection onto a weighted L1-ball
    parameter radius: the radius of the L1-ball
    parameter weights: the weight
    parameter y: the point to be projected
    """

    y_sign = np.sign(y)
    y_proj = y_sign * y
    act_ind = range(len(y))
    while True:
        # calculate y_bar_act
        y_proj_act = y_proj[act_ind]
        weights_act = weights[act_ind]

        x_sol_hyper, lamb = hyperplane(y_proj_act, weights_act, radius)
        y_proj_act = np.maximum(x_sol_hyper, 0.0)
        y_proj[act_ind] = y_proj_act

        # update the active set
        act_ind = y_proj > 0

        inact_ind_cardinality = sum(x_sol_hyper < 0)
        if inact_ind_cardinality == 0:
            x_opt = y_proj * y_sign
            break

    return x_opt, lamb


def Frank_Wolfe_Lp(A, x, y, mu, p, radius):
    """
    Solve the signal recovery problem as
    min 0.5 * ||Ax - y||_2^2
    s.t. ||x||_p^p <= r,
    where 0 < p < 1.
    parameter radius: the radius of the Lp-ball
    parameter mu: the step-size of the projected gradient descent method
    parameter y: the measurement of signal
    parameter A: the measurement matrix
    """

    eta = 0.8               # the ratio of step-size in Frank-Wolfe method
    bisection_flag = 0      # the flag of whether bisection method succeeds
    tol = 1e-8              # tolerance of numerical precision
    iter = 0                # the number of total iteration
    iter_proj = 0           # the number of the call of projected gradient descent method
    iter_fw = 0             # the number of the call of Frank-Wolfe method
    stopping_tol = 1e-6     # the tolerance of stopping criteria

    while True:
        iter += 1
        grad = np.dot(A.T, A.dot(x) - y)    # compute the gradient of objective
        # Check whether the current point is on the boundary of the Lp-ball
        if abs(LA.norm(x, p) ** p - radius) <= tol:
            """
            If the current point is on the boundary of the Lp-ball, approximate the Lp-ball as
            a weighted L1-ball and apply projected gradient method.
            """
            iter_proj += 1
            x_pre = x
            z = x - mu * grad
            act_ind = np.where(abs(x) > tol)[0]
            for i in range(len(z)):
                if i not in act_ind:
                    z[i] = 0
            w = np.zeros(len(x))
            w[act_ind] = p * abs(x[act_ind]) ** (p - 1)
            radius_L1 = radius - LA.norm(x[act_ind], p) ** p + w[act_ind].dot(abs(x[act_ind]))

            # Do Re-weighted L1 projection
            x, lamb = WeightedL1Projection(z, w, radius_L1)

            # Stopping criteria
            proj_res = LA.norm(x - x_pre, 2)
            print('{:5d}    {}:   Obj = {:3.3f}   Res = {:4.3e}   #nonzero = {}'.format(iter, 'Boundary case', loss(A, x, y), proj_res, len(np.nonzero(x)[0])))
            if proj_res < stopping_tol:
                break

        elif LA.norm(x, p) ** p < radius:
            """
            If the current point is inside the Lp-ball, apply Frank-Wolfe method.
            """
            iter_fw += 1
            grad_abs = abs(grad)
            max_ind = np.argmax(grad_abs)

            # Update the transformed variable
            z = np.zeros(len(y))
            z[max_ind] = radius

            # Update the original variable
            s_abs = z ** (1 / p)
            s = - s_abs * np.sign(grad)
            d = s - x
            print('{:5d}    {}:   Obj = {:3.3f}   Res = {:4.3e}   #nonzero = {}'.format(iter, 'Interior case', loss(A, x, y), grad.dot(-d), len(np.nonzero(x)[0])))
            if grad.dot(-d) < stopping_tol:
                break

            # Determine whether find a gamma_bar such that the new iterate is on the boundary
            gamma_amijo = - 2 * (1 - eta) * grad.dot(d) / (LA.norm(A.dot(d), 2) ** 2)
            if LA.norm(x + gamma_amijo * d, p) ** p > radius:
                g_r = gamma_amijo
                g_l = 0.0
                gamma_bar = (g_l + g_r) / 2.0
                res = LA.norm(x + gamma_bar * d, p) ** p - radius
                while abs(res) > tol:
                    if g_r - g_l <= 0:
                        bisection_flag = 1
                        break
                    if res > 0:
                        g_r = gamma_bar
                    else:
                        g_l = gamma_bar
                    gamma_bar = (g_l + g_r) / 2.0
                    res = LA.norm(x + gamma_bar * d, p) ** p - radius
            else:
                gamma_bar = 1.0

            if bisection_flag == 1:
                print('Can not find a root by bisection method!')
                print(LA.norm(x + gamma_bar * d, p) ** p - radius)
                break

            # Get step-size for Frank-Wolfe method
            gamma = min(1, gamma_bar, gamma_amijo)
            x += gamma * d

        else:
            print('The new iterate is infeasible!')

    return x, iter_fw, iter_proj
