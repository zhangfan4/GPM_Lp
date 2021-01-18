"""
    This file is the implementation of the Projectin gradient method
    See the Algorithm Framework for details

"""


import numpy as np
import sys
from numpy import linalg as LA
import Lp_proj
import simplex_RT
<<<<<<< HEAD
from auxiliary_projection import projection_onto_hyperplane, projection_onto_weighted_l1_norm_ball
=======
import inexact_proj
>>>>>>> f24af5513d4e374ca8984e0694265f8a0aaae32a


def loss(A, x, y):
    """
    the objective function value
    return 0.5 * ||Ax - y||_2^2
    """
    return 0.5 * LA.norm(A.dot(x) - y, 2) ** 2


def gradient_of_loss(A, x, y):
    """
    the gradient of the objective function
    return A^T(Ax - y)
    """
    return A.T.dot(A.dot(x) - y)




def GPM(A, x, y, step_size, p, radius):
    """
    Gradient Projection Method
    :parameter A: the measurement matrix
    :parameter x: the initial guess of the signal 
    :parameter y: the measurement of signal
    :parameter step_size: the step-size of the GPM
    :parameter radius: the radius of the L1-ball
    :returns: x: output of the GPM, 
              iter_Lp: # the projection onto the Lp-norm ball, 
              iter_L1: # the projection onto the Weighed L1-norm ball
    """

    precision = 1e-6              # tolerance of numerical precision
    iter = 0                # the number of total iteration
    iter_L1 = 0           # the number of the call of projected gradient descent method
    iter_Lp = 0             # the number of the call of Frank-Wolfe method
    stopping_tol = 1e-6     # the tolerance of stopping criteria

    while True:
        iter += 1

        # compute the gradient of objective function
        grad = gradient_of_loss(A, x, y)    

        # Check whether the current point is on the boundary of the Lp-ball
        if abs(LA.norm(x, p) ** p - radius) <= precision:
            """
            Case I: x_k is on the boundary of the Lp-ball, ||x_k||_p^p = r
            """
            iter_L1 += 1
            x_pre = x

            # the point to be projected
            z = x - step_size * grad

            # Check z feasible or infeasible (necessary?)
            if LA.norm(z, p) ** p <= radius + precision:  # CASE I.1: z is feasible, that is, z is in the lp ball
                x = z

            else:                        # CASE I.2: z is infeasible, that is, z is outside the lp ball, we need do projection
                ''' Projecting z onto the weighed L1 norm ball '''
                # print('outer-inner')

                # keep the same active set between x and z
                active_index = np.where(abs(x) > precision)[0]
                for i in range(len(z)):
                    if i not in active_index:
                        z[i] = 0

                # weight vector
                w = np.zeros(len(x))
                w[active_index] = abs(x[active_index]) ** (p - 1)

                # radius
                radius_L1 = w[active_index].dot(abs(x[active_index]))
                # weighted L1 projection
                x = projection_onto_weighted_l1_norm_ball(z, w, radius_L1)


            # Stopping criteria
            proj_residual = LA.norm(x - x_pre, 2)   # ||x_new - x_old||_2
            if proj_residual < stopping_tol:
                break

        elif LA.norm(x, p) ** p < radius - precision:
            """
            Case II: x_k is inside the Lp-ball, ||x_k||_p^p < r
            """
            iter_Lp += 1

            # Termination criterion \|\nabla f(x_k)\|_2 <= eps_2
            if LA.norm(grad, 2) < stopping_tol:
                break

<<<<<<< HEAD
            z = x - step_size * grad

            if LA.norm(z, p) ** p <= radius + precision:    # CASE II.1: z is feasible, that is, z is in the lp ball
                # print('inner-inner')
                x = z

            else:               # CASE II.2: z is infeasible, that is, z is outside the lp ball, we need do projection
                dim = len(z)
                x_init = x  # the initial points for the algorithm 1

                # Generate epsilon according to x.
                # see 2101.01350.pdf, section 5.1 for details
                epsilon = 0.9 * (1. / dim * (radius - LA.norm(x, p) ** p)) ** (1. / p) * np.ones(dim)  
                # ensure that the point is feasible.
=======
            z = x - alpha * grad
            if (LA.norm(z, p) ** p - radius) <= tol:
                
                # print('inner-inner')
                x = z

            elif LA.norm(z, p) ** p > radius:
                ''' Projecting z onto the Lp norm ball '''
                # print('inner-outer')

#                 dim = len(z)
#                 x_ini = x  # the initial points for the algorithm

                # %% Generate epsilon according to x.
#                 epsilon = 0.9 * (1. / dim * (radius - LA.norm(x, p) ** p)) ** (1. / p) * np.ones(
#                     dim)  # ensure that the point is feasible.
#                 x = Lp_proj.WeightLpBallProjection(dim, x_ini, z, p, radius, epsilon)
                
                beta = inexact_proj.bisection(x, d, radius, p, tol)  # line search for the stepsize beta satisfying ||x+beta*d||_p^p = radius
                x += beta * d
>>>>>>> f24af5513d4e374ca8984e0694265f8a0aaae32a

                x = Lp_proj.WeightLpBallProjection(dim, x_init, z, p, radius, epsilon)
                
        else:
            print('The new iterate is infeasible!')

    return x
