import numpy as np
from numpy import linalg as LA
from scipy.optimize import root_scalar


def fun_eval(x, d, mid, p, gamma):
    """
        Function evaluation
        f(lam) = ||x + beta * d||_p^p - gamma
    """
    x_temp = x + mid * d
    f_val = LA.norm(x_temp, p) ** p - gamma
    return f_val


def bisection(x_ini, d, gamma, p, tol):
    """
        Bisection method for root finding
        :parameter lam: variable
        :parameter y_proj: the point to be projected
        :parameter gamma: radius
        :parameter w: weights
        :returns: x_sub: the projection onto the hyperplane 
                  lamda: root
    """

    low = 0.
    up = 1.
    mid = 0.5

    '''Plan A: handmade root finding'''

    while True:
        mid_val = fun_eval(x_ini, d, mid, p, gamma)
        # print('+'*40)
        # print(low, fun_eval(low, w, y_proj, gamma))
        if abs(mid_val) <= tol:
            break
        elif mid_val < 0:
            low = mid
        else:
            up = mid

        # if (up - low) <= tol ** 2:
        #     print('Fail to find the root!')
        #     break

        mid = (low + up) / 2
    beta = mid

    # '''Plan B: using package'''
    # def fun_eval(beta):
    #     """
    #         Function evaluation
    #         f(lam) = sum_i w_i max(y_i - lam w_i, 0) - gamma
    #     """
    #     return LA.norm(x_ini + beta * d, p) ** p - gamma
    # 
    # sol = root_scalar(fun_eval, method='toms748', bracket=[low, up], xtol=tol)
    # beta = sol.root
    # 
    # print('residual:', LA.norm(x_ini + beta * d, p) ** p - gamma)

    return beta
