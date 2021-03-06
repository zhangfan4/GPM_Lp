#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 14:37:41 2021

@author: moumatsu
"""


import numpy as np
import numpy.linalg as LA

import root_finding


def proj_simplex(v, s=1):
    """ Compute the Euclidean projection on a positive simplex
    Solves the optimisation problem (using the algorithm from [1]):
        min_w 0.5 * || w - v ||_2^2 , s.t. \sum_i w_i = s, w_i >= 0 
    Parameters
    ----------
    v: (n,) numpy array,
       n-dimensional vector to project
    s: int, optional, default: 1,
       radius of the simplex
    Returns
    -------
    w: (n,) numpy array,
       Euclidean projection of v on the simplex
    Notes
    -----
    The complexity of this algorithm is in O(n log(n)) as it involves sorting v.
    Better alternatives exist for high-dimensional sparse vectors (cf. [1])
    However, this implementation still easily scales to millions of dimensions.
    References
    ----------
    [1] Efficient Projections onto the .1-Ball for Learning in High Dimensions
        John Duchi, Shai Shalev-Shwartz, Yoram Singer, and Tushar Chandra.
        International Conference on Machine Learning (ICML 2008)
        http://www.cs.berkeley.edu/~jduchi/projects/DuchiSiShCh08.pdf
    """
    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
    n, = v.shape  # will raise ValueError if v is not 1-D
    # check if we are already on the simplex
    if v.sum() == s and np.alltrue(v >= 0):
        # best projection: itself!
        return v
    # get the array of cumulative sums of a sorted (decreasing) copy of v
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    # get the number of > 0 components of the optimal solution
    rho = np.nonzero(u * np.arange(1, n+1) > (cssv - s))[0][-1]
    # compute the Lagrange multiplier associated to the simplex constraint
    theta = float(cssv[rho] - s) / rho
    # compute the projection by thresholding v using theta
    w = (v - theta).clip(min=0)
    return w


def proj_l1ball(v, radius=1):
    """ Compute the Euclidean projection on a L1-ball
    Solves the optimisation problem (using the algorithm from [1]):
        min_w 0.5 * || w - v ||_2^2 , s.t. || w ||_1 <= radius
    Parameters
    ----------
    v: (n,) numpy array,
       n-dimensional vector to project
    radius: int, optional, default: 1,
       radius of the L1-ball
    Returns
    -------
    w: (n,) numpy array,
       Euclidean projection of v on the L1-ball
    Notes
    -----
    Solves the problem by a reduction to the positive simplex case
    See also
    --------
    euclidean_proj_simplex
    """
    assert radius > 0, "Radius s must be strictly positive (%d <= 0)" % radius
    n, = v.shape  # will raise ValueError if v is not 1-D
    # compute the vector of absolute values
    u = np.abs(v)
    # check if v is already a solution
    if u.sum() <= radius:
        # L1-norm is <= s
        return v
    # v is not already a solution: optimum lies on the boundary (norm == s)
    # project *u* on the simplex
    w = proj_simplex(u, radius)
    # compute the solution to the original problem on v
    w *= np.sign(v)
    return w


def proj_hyperplane(y, weight, radius):
    """
    Projection onto the weigthed l1 norm ball:
        min_{x} 0.5 * ||x-y||_2^2
        s.t.    \sum_i weight_ix_i = radius
    Parameters
    ----------
    y: (n, ) numpy array,
       n-dimensional vector to project
    weight: (n, ) numpy array,
       n-dimensional weight vector for l1 norm ball
    radius: int, optional, default: 1,
       radius of the weighted L1-ball
    Returns
    -------
    x_opt: (n, ) numpy array,
       Euclidean projection of y onto the hyperplane defined 
       byweight and radius
    """
    # initial guess for bisection method
    lamb = (weight.dot(y) - radius) / LA.norm(weight)**2
    x_proj = y - lamb * weight
    
    return x_proj, lamb


def proj_weightedl1ball(y, weight, radius, lamb, method):
    """
    Projection onto the weigthed l1 norm ball:
        min_{x} ||x-y||_2^2
        s.t.    \sum_i weight_ix_i\leq radius
                x_i >= 0
        where y_i >= 0, i=1,...,n
    Parameters
    ----------
    y: (n, ) numpy array,  all elements positive
       n-dimensional vector to project
    weight: (n, ) numpy array, all elements positive
       n-dimensional weight vector for l1 norm ball
    radius: int, optional, default: 1,
       radius of the weighted L1-ball
    Returns
    -------
    x_opt: (n, ) numpy array,  all elements non-negative
       Euclidean projection of y onto the weighted L1-ball
    """
    assert all(entry >= 0 for entry in y), "y must be positive!"

    if weight.dot(y) <= radius:
        return y, 0

    else:
        # use the projection way
        if method == 'bisection':
            x_opt, lamb = root_finding.bisection(weight,
                                                 y, 
                                                 radius, 
                                                 lamb)
        elif method == 'projection':
            
            act_ind = range(len(y))
            while True:
                # calculate y_bar_act
                y_act = y[act_ind]
                weights_act = weight[act_ind]
        
                x_sol_hyper, lamb = proj_hyperplane(y_act, 
                                                    weights_act, 
                                                    radius)
                y_act = np.maximum(x_sol_hyper, 0.0)
                y[act_ind] = y_act
        
                # update the active set
                act_ind = y > 0
        
                if sum(x_sol_hyper < 0) == 0:
                    x_opt = y * np.sign(y)
                    break

    return x_opt, lamb


def proj_lpball(y, p, x, radius=1):
    """
    Projection onto the Lp norm ball:
        min_{x} ||x-y||_2^2
        s.t.    ||x||_p^p <= radius
    Parameters
    ----------
    y: (n, ) numpy array,
       n-dimensional vector to project
    p: a positive scalar,
       0 < p < 1, the Lp norm
    x: initial guess for the projection point
    radius: int, optional, default: 1,
       radius of the Lp norm ball
    Returns
    -------
    x_opt: (n, ) numpy array,
       Euclidean projection of y onto the weighted L1-ball
    """
    # configuration
    n, = x.shape
    tau = 1.1
    M = 1e4
    MAX_ITERS = 1e3
    delta_tol = 1e-10
    epsilon = np.ones(n) 
    epsilon *= 0.9 * (1./n * (radius - LA.norm(x, p)**p))**(1./p)
    nonzero_num = np.count_nonzero(y)
    
    # preprocess
    n = len(y)
    
    iter = 0
    lamb = 0
    
    if LA.norm(y, p) ** p <= radius:
        print("y_bar is inside the Lp norm ball!")
        return y, nonzero_num
    
    while True:
        iter += 1
        # step 3: compute the weight
        weight = p * (np.abs(x) + epsilon) ** (p - 1)
        
        # step 4: solve the subproblem (9)
        radius_k = radius - LA.norm(np.abs(x) + epsilon, p)**p + \
            weight.dot(np.abs(x))
        
        assert radius > 0, "Radius must be strictly positive (%d <= 0)" % \
            radius_k
        
        # solve the subproblem via root finding
        # x_opt, lamb = root_finding.bisection(lamb, weight,
        #                                     abs(y), radius)
        x_opt, lamb = proj_weightedl1ball(np.abs(y), weight, 
                                          radius_k, lamb,
                                          'bisection')
        
        nonzero_num = np.count_nonzero(x_opt)
        
        x_real = x_opt * np.sign(y)
        
        reduction = x_real - x
        condition = (LA.norm(reduction) * LA.norm(weight,
                     2)**tau <= M) 
        
        error = abs(LA.norm(x_real, p)**p - radius)
        
        # update epsilon
        if condition and min(epsilon) >= 1e-20:
            epsilon *= np.minimum(error, 1/np.sqrt(iter))
        
        if iter == 1:
            alpha_res_init = (1/n) * np.sum(np.abs((np.abs(y) - x_opt) * 
                                        x_opt - p * lamb *
                                        x_opt** p))
            beta_res_init = (1/n) * np.abs(LA.norm(x_opt, p) ** p - 
                                        radius)
        alpha_res = (1/n) * np.sum(np.abs((np.abs(y) - x_opt) * 
                                    x_opt - p * lamb *
                                    x_opt**p))
        beta_res = (1/n) * abs(LA.norm(x_opt, p) ** p - 
                                    radius)
        
        # print("interior_iter=%3d, error=%3.3e, nonzero=%3.3e, obj=%3.3e" % (iter, beta_res, nonzeros_num, obj_value))
        
        # step 6: 
        x = x_real
        
        # stopping criterion
        if max(alpha_res, beta_res) <= delta_tol * \
            max(alpha_res_init, beta_res_init, 1):
            break              
        
        if iter >= MAX_ITERS:  
            print("The solution is not so good")
            break
        
    return x_real, nonzero_num
        
        
        
        
        
        
        
        
    
    
    
    
    