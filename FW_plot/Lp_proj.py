#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 14:49:51 2020

@author: Xiangyu Yang
"""

import numpy as np
import sys
from numpy import linalg as LA
import simplex_RT
# from scipy import optimize
# import random
# import matplotlib.pyplot as plt


# def point_Projected(a,N):
#     """
#     Generate diffenent types of points with i.i.d. random Guassian distribution.
#     :Type I: All component follow N(0,1)
        
#     :Type II: 7/8 components of y are i.i.d random Guassian numbers of mean 0 and standard deviation 0.2, the rest are i.i.d random Gaussian numbers of
#     mean 0.9 and standard deviation 0.2.
    
#     :Type III: 
        
#     """ 
#     ## %% Generate Type I
#     mu, sigma = a/N, 1e-3
#     p = np.random.normal(mu,sigma,N)
#     return p


#%% Compuite the objective
def loss(x, y):
    """
    Parameterss
    ----------
    x : the current solution of dimension n by 1.
    
    y : The given point to be projected

    Returns
    -------
    solution : the solution to the lp projection
    """
    obj = 0.5 * LA.norm(x-y, ord=2) ** 2
    return obj
#%% the start of the inner loop
def hyperplane(y_proj_act, weights_act, gamma_k): 
    # %% Projection onto hyperplane
    """
    Parameters
    ----------
    y_proj_act : being projected point with active index
    
    weights_act : the corresponding weights with the same active index as y_proj
    
    gamma_k : radius of weighted L1 ball of the k-th subproblem

    Returns
    -------
    w : the solution to the weighted l1 projection

    """
    
    scalar1 = np.dot(weights_act, y_proj_act) - gamma_k
    scalar2 = LA.norm(weights_act, ord=2) ** 2
    try:
        scalar3 = np.divide(scalar1, scalar2)
    except ZeroDivisionError:
        print("Error! - derivation zero for scalar2 =", scalar2)
        sys.exit(1)
    x_sub = y_proj_act - scalar3 * weights_act
    return x_sub, scalar3


def WeightSimplexProjection(n, y, signum, gamma_k, weights):
    """
    Parameters
    ----------
    n: dimension
    y: the point to be projected 
    signum: An array that record the sign of the y
    gamma_k: the radius of the weighted l1 ball for the k-th subproblem
    weights: the fixed weights for the k-th subproblem
    
    Returns
    -------
    x_opt: the solution to the k-th subproblem
    
    """
    y_proj = signum * y # elementwise multiplication of two ndarray : the datapoint to be projected
    act_ind = list(range(n)) # record active index
    while True: #Because this algorithm terminates in finite steps
    # calculate y_bar_act            
        y_proj_act = y_proj[act_ind]
        weights_act = weights[act_ind]

        x_sol_hyper, lambd = hyperplane(y_proj_act, weights_act, gamma_k)  # projection onto the hyperplane

        #%% dimension reduction and now the y_bar_act is the next projection point. Projection onto the non-negative orthont.  
        #Once the non-negative components are detected, then elemiate them. These components are kept as 0 during next projection. 
        y_proj_act = np.maximum(x_sol_hyper, 0.0)
        y_proj[act_ind] = y_proj_act # back to the initial index
        
        #%% update the active set
        act_ind = []
        
        #%% We only need to find the nonzeros and then extract its index.
        arr_nonzero_y_proj = np.nonzero(y_proj>0)
        arr_nonzero_y_proj = arr_nonzero_y_proj[0]
        act_ind = arr_nonzero_y_proj.tolist()
        
        signum_x_inner = np.sign(x_sol_hyper)
        inact_ind_cardinality = sum(elements < 0 for elements in signum_x_inner) #inact_ind
        
        if inact_ind_cardinality == 0:
            x_opt = y_proj
            break

    return x_opt, lambd
#%%==========Outer Loop=======

def WeightLpBallProjection(n, x, y, p, radius, epsilon):
    """
    Lp ball exact projection
    param radius: Lp ball radius.
    param y_bar: parameter vector of dimension n by 1 to be projected
    return: projection of theta onto lp ball ||x||_p <= radius, and the iteration k

    """
#%% Input parameters
    Tau, tol = 1.1, 1e-8
    Iter_max = 1e3
    M = 1e4
    eps_inf = 1e-24 * np.ones(n)
        
    # record the signum of the point to be projected to restore the solution
    signum = np.sign(y) 
    x_final = np.zeros(n)
                  
    bar_y = signum * y  # Point lives in the positive orthant
    
    # store the  values
    res_alpha = []  # residual_alpha
    res_beta = []  # residual_beta
    res_iterate = []  # x iterates
    res_nonzero = []
    
    epsilon_seq = []
    lambda_opt_seq = []
    gamma_k_list = []
    weights_seq = []
    
    
    count = 0 # the iteration counter
    counter = 0 # count the number of times the condition is triggered
# %% Subproblem solve: exact projection onto the weighted simplex 

    if LA.norm(y, p) ** p <= radius: #  Determine the current ball whether it falls into the lp-ball.
        print('The current point falls into the Lp ball. Please choose another new point!')
        return None
    
    else: 
        Flag_gamma_pos = 'Success'
        lam = 0  # initial value of lambda
        while True:
            count = count + 1
            # Step 3 of algorithm1: Reweighing: Compute the weights
            # Typo in original code!
            if count == 1:
                x = abs(x)

            weights = p * (abs(x) + epsilon) ** (p-1)
            weights_seq += [weights]
            
            # Step 4 of algorithm1: Subproblem solving
            gamma_k = radius - LA.norm(x+epsilon,p) ** p + np.dot(weights, x)  # Typo in original code 'np.abs(x)'!
            # print(radius - LA.norm(np.abs(x)+epsilon,p) ** p)
            # print(gamma_k)
            # print('-'*20)
            gamma_k_list += [gamma_k]
            
            if gamma_k <= 0:
                Flag_gamma_pos = 'Fail'
                print('The current Gamma is not positive!!!')
                break
                
            #%% Calling algorithm2: weighted l1 ball projection
            x_opt, lam = WeightSimplexProjection(n, y, signum, gamma_k, weights)  # x_opt: R^n
            # x_opt, lam = simplex_RT.bisection(signum, lam, y, gamma_k, weights)  # x_opt: R^n
            # print(lam)

            num_nonzeros = np.count_nonzero(x_opt)
            res_nonzero += [num_nonzeros]
            
            #%% Compute the Objective value
            x_tem = signum * x_opt
            obj_k = loss(x_tem, y)
            
            #%% whether the update condition is triggerd for epsilon

            local_reduction = x_opt - x    # in the limit, it should be zero
            local_reduction_norm = LA.norm(local_reduction, ord=2)

            # Adapted by our current paper.
            sign_weight = np.sign(local_reduction) * weights   # Typo in original code!
            condition_left = local_reduction_norm * LA.norm(sign_weight, ord=2) ** Tau
            condition_right = M
            

            #%% Determine whether to trigger the update condition            
            error_appro = np.abs(LA.norm(x_opt, p)**p - radius)
            epsilon_seq += [epsilon]
            
            if condition_left <= condition_right:
                epsilon = np.maximum(np.minimum(error_appro, 1/(np.sqrt(count))) * epsilon, eps_inf)
                counter = counter + 1
                # print()
                
            eps_norm = LA.norm(epsilon, np.inf)
                        
            #%% Checking the termination conditon
            
            act_ind_outer = []  # collect the active index from the (k+1)-th solution
            for ind in range(len(x_opt)):
                if x_opt[ind] > 0:
                    act_ind_outer.append(ind)
                   
            # Determine the inactive set whether remains unchanged
            #%% Determine whether this collection is empty
            if act_ind_outer:  # Nonempty inactive set I. Our lemma shows the I(x^k) is nonempty

                # Begin to calculate the residual
                y_act_ind_outer = bar_y[act_ind_outer]
                x_act_ind_outer = x_opt[act_ind_outer] #(k+1)th iterate
                weights_ind_outer = weights[act_ind_outer]
            
                ## Check this formula
                #%% Compute multiplier lambda and tow residuals
                lambda_opt = np.divide(np.sum(y_act_ind_outer - x_act_ind_outer), np.sum(weights_ind_outer))
                lambda_opt_seq += [lambda_opt]

                # Typo in original code! (no 1/n in paper)
                residual_alpha = (1/n) * np.sum(np.abs((bar_y - x_opt) * x_opt - p * lambda_opt * x_opt ** p))
                residual_beta = (1/n) * error_appro

                res_alpha += [residual_alpha]
                res_beta += [residual_beta]

                # Step 6 of our algorithm: go to the (k+1)-th iterate.
                x = x_opt          # (k+1)-th solution
                res_iterate += [x]  # store each iterate

                print('{0:3d}: Obj = {1:4.3e}, alpha = {2:4.3e}, beta = {3:4.3e}, eps = {4:4.3e}, dual = {5:4.3e}, #nonzeros = {6:2d}, x-x_p = {7:4.3e}'.format(count, obj_k, residual_alpha, residual_beta, eps_norm, lambda_opt, num_nonzeros, local_reduction_norm), end=' ')
                print()
                # Check the stopping criteria
                if np.maximum(residual_alpha, residual_beta) <= tol * np.max([res_alpha[0], res_beta[0], 1]) or count >= Iter_max:
                    if count >= Iter_max:
                        Flag_gamma_pos = 'Fail'
                        save_flag = True
                        print("The solution is not so good")
                        break
                    else:
                        Flag_gamma_pos = 'Success'
                        x_final = signum * x_opt  # element-wise product
                        save_flag = False
                        break

    return x_final, save_flag

