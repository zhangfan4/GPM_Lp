import numpy as np
from scipy.optimize import root_scalar


def fun_eval(lam, w, y, gamma):
    """
        Function evaluation
        f(lam) = sum_i w_i max(y_i - lam w_i, 0) - gamma
    """
    x_temp = y - lam * w
    f_val = np.dot(w.T, np.maximum(x_temp, 0)) - gamma
    return f_val


def bisection(signum, lam, y_proj, gamma, w):
    """
        Bisection method for root finding
        :parameter lam: variable
        :parameter y_proj: the point to be projected
        :parameter gamma: radius
        :parameter w: weights
        :returns: x_sub: the projection onto the hyperplane 
                  lamda: root
    """
    y_proj = y_proj * signum  # elementwise multiplication of two ndarray : the datapoint to be projected

    low = 0
    up = max(y_proj/w)
    mid = lam
    tol = 1e-10

    # def fun_eval(lam):
    #     """
    #         Function evaluation
    #         f(lam) = sum_i w_i max(y_i - lam w_i, 0) - gamma
    #     """
    #     return np.dot(w.T, np.maximum(y_proj - lam * w, 0)) - gamma
    #
    # sol = root_scalar(fun_eval, x0=lam, method='toms748', bracket=[low, up], xtol=tol)
    # lamda = sol.root

    while True:
        mid_val = fun_eval(mid, w, y_proj, gamma)
        # print(mid, mid_val)
        # print('+'*40)
        # print(low, fun_eval(low, w, y_proj, gamma))
        if np.abs(mid_val) <= tol:
            break
        elif mid_val < 0:
            up = mid
        else:
            low = mid

        if (up - low) <= tol ** 2:
            print('Fail to find the root!')
            break

        mid = (low + up) / 2

    lamda = mid
    x_sub = np.maximum(y_proj - lamda * w, 0)
    return x_sub, lamda
