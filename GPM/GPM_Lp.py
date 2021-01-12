import numpy as np
import sys
from numpy import linalg as LA
import Lp_proj
import simplex_RT


def loss(A, x, y):
    """
    return 0.5 * ||Ax - y||_2^2
    """
    return 0.5 * LA.norm(A.dot(x) - y, 2) ** 2


def hyperplane(y_proj_act, weights_act, gamma_k):
    """
    Do projection onto a hyperplane
    min 0.5 * ||x_sub - y_proj_act||_2^2
    s.t. y_proj_act^T weights_act = r,
    :parameter gamma_k: the radius of L1-ball
    :parameter y_proj_act: the point to be projected
    :returns: x_sub: the projection onto the hyperplane 
              scalar3: x_sub = y_proj_act - scalar3 * weights_act
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
    min 0.5 * ||x_opt - y||_2^2
    s.t. weights^T |y| = radius,
    :parameter radius: the radius of the L1-ball
    :parameter weights: the weight
    :parameter y: the point to be projected
    :return: x_opt: primal optimal 
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

    return x_opt


def GPM(A, x, y, alpha, p, radius):
    """
    Gradient Projection Method
    :parameter A: the measurement matrix
    :parameter x: the initial guess of the signal 
    :parameter y: the measurement of signal
    :parameter alpha: the step-size of the GPM
    :parameter radius: the radius of the L1-ball
    :returns: x: output of the GPM, 
              iter_Lp: # the projection onto the Lp-norm ball, 
              iter_L1: # the projection onto the Weighed L1-norm ball
    """

    eta = 0.8               # the ratio of step-size in Frank-Wolfe method
    bisection_flag = 0      # the flag of whether bisection method succeeds
    tol = 1e-6              # tolerance of numerical precision
    iter = 0                # the number of total iteration
    iter_L1 = 0           # the number of the call of projected gradient descent method
    iter_Lp = 0             # the number of the call of Frank-Wolfe method
    stopping_tol = 1e-6     # the tolerance of stopping criteria

    while True:
        iter += 1
        grad = np.dot(A.T, A.dot(x) - y)    # compute the gradient of objective
        # x_pnorm = LA.norm(x, p) ** p

        # Check whether the current point is on the boundary of the Lp-ball
        if abs(LA.norm(x, p) ** p - radius) <= tol:
            """
            Case I: x_k is on the boundary of the Lp-ball
            """
            iter_L1 += 1
            x_pre = x
            z = x - alpha * grad

            # Check z feasible or infeasible (necessary?)
            if abs(LA.norm(z, p) ** p - radius) <= tol:  # Will this situation happen?
                print('outer-outer')
                x = z

            else:
                ''' Projecting z onto the weighed L1 norm ball '''
                print('outer-inner')
                # find the active set and calculate the weight
                act_ind = np.where(abs(x) > tol)[0]
                for i in range(len(z)):
                    if i not in act_ind:
                        z[i] = 0
                w = np.zeros(len(x))
                w[act_ind] = p * abs(x[act_ind]) ** (p - 1)
                radius_L1 = w[act_ind].dot(abs(x[act_ind]))
                # weighted L1 projection
                x = WeightedL1Projection(z, w, radius_L1)

                # # TODO: zero element encountered, Insight: select inactive indices
                # signum = np.ones(len(act_ind))
                # lam = 0
                # x_proj, _ = simplex_RT.bisection(signum, lam, z[act_ind], radius_L1, w[act_ind])
                # x = np.zeros_like(z)
                # x[act_ind] = x_proj


            # Stopping criteria
            proj_res = LA.norm(x - x_pre, 2)
            print('{:5d}    {}:   Obj = {:3.3f}   Res = {:4.3e}   #nonzero = {}'.format(iter, 'Boundary case',
                                                                        loss(A, x, y), proj_res, np.count_nonzero(x)))
            # Termination criterion \|x_k - x_{k-1}\|_2 <= eps_1
            if proj_res < stopping_tol:
                break

        elif LA.norm(x, p) ** p < radius:
            """
            Case II: x_k is inside the Lp-ball
            """
            iter_Lp += 1
            norm_grad = alpha ** 2 * LA.norm(grad) ** 2

            # Termination criterion \|\nabla f(x_k)\|_2 <= eps_2
            if norm_grad < stopping_tol:
                break

            z = x - alpha * grad
            if LA.norm(z, p) ** p > radius:
                ''' Projecting z onto the Lp norm ball '''
                print('inner-outer')
                dim = len(z)
                x_ini = x  # the initial points for the algorithm

                # %% Generate epsilon according to x.
                epsilon = 0.9 * (1./dim * (radius - LA.norm(x, p) ** p)) ** (1./p) * np.ones(dim)  # ensure that the point is feasible.
                x = Lp_proj.WeightLpBallProjection(dim, x_ini, z, p, radius, epsilon)

            else:
                print('inner-inner')
                x = z

            print('{:5d}    {}:   Obj = {:3.3f}   Res = {:4.3e}   #nonzero = {}'.format(iter, 'Interior case',
                                                                                        loss(A, x, y), norm_grad,
                                                                                        np.count_nonzero(x)))
        else:
            print('The new iterate is infeasible!')

    return x, iter_Lp, iter_L1
