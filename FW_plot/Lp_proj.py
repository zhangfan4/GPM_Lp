
import numpy as np
import sys
from numpy import linalg as LA
import simplex_RT
from auxiliary_projection import *


def loss(x, y):
    """
    Compute the objective function values 

    Parameterss
    ----------
    x : the current solution of dimension n by 1.
    
    y : The given point to be projected

    Returns
    -------
    solution : the solution to the lp projection
    """
    obj = 0.5 * LA.norm(x - y, 2) ** 2
    return obj

def print_iteration_info(iter, objective_value,residual_alpha, residual_beta, eps_norm, lam, num_nonzeros, local_reduction_norm, gamma):
    print('Iter{0:3d}: '.format(iter), end=' ')
    print('Obj = {0:4.3e},'.format(objective_value), end=' ')
    print('alpha = {0:4.3e},'.format(residual_alpha), end=' ')
    print('beta = {0:4.3e},'.format(residual_beta), end=' ')
    print('eps_inf_norm = {0:4.3e},'.format(eps_norm), end=' ')
    print('dual = {0:4.3e},'.format(lam), end=' ')
    print('#nonzeros = {0:2d},'.format(num_nonzeros), end=' ')
    print('x-x_p = {0:4.3e},'.format(local_reduction_norm), end=' ')
    print('gamma = {0:4.3e},'.format(gamma), end=' ')
    print()



def WeightLpBallProjection(n, x, y, p, radius, epsilon):
    """
    Lp ball exact projection
    param n: the length of the vector x and y
    param radius: Lp ball radius.
    param y_bar: parameter vector of dimension n by 1 to be projected
    return: projection of theta onto lp ball ||x||_p <= radius, and the iteration k

    """

    # Input parameters for Algorithm 1
    Tau, tol = 1.1, 1e-6
    Iter_max = 1e3
    M = 1e4
        
    # record the sign of y of the point to be projected to restore the solution
    sign_of_y = np.sign(y) 

    # the final solution returned by the Algorithm 1
    x_final = np.zeros(n)
                  
    y_bar = sign_of_y * y  # Point lives in the positive orthant
    
    # store the  information of iterations

    # residual_alpha, residual_beta are defined in Eq(39) and Eq(25)
    res_alpha = []  # residual_alpha
    res_beta = []  # residual_beta

    # store the iteration point
    x_iterate = []  

    # number of non-zero elements
    res_nonzero = []  
    
    # store the epsilon
    epsilon_seq = []


    lambda_seq = []
    gamma_k_list = []
    weights_seq = []
    
    
    iter = 0 # the iteration counter

    # count the number of times the condition Eq(9) is triggered
    count_trigger = 0 

    # Subproblem solve: exact projection onto the weighted simplex 
    if LA.norm(y, p) ** p <= radius: #  Determine the current ball whether it falls into the lp-ball.
        print('The current point falls into the Lp ball. Please choose another new point!')
        return None
    

    Flag_gamma_pos = 'Success'


    ###########  Algorithm 1 ########################
    lam = 0  # initial value of lambda
    while True:
        iter += 1
        
        # each entry of x should have the same sign as the corresponding entry of y
        if iter == 1:
            x = x * sign_of_y

        # Step 3 of Algorithm 1: Compute the weights
        weights = p * (np.abs(x) + epsilon) ** (p-1)
        weights_seq += [weights]
        
        # Step 4 of algorithm1: Subproblem solving, Eq(8) for the definition of gamma_r and weights
        gamma_k = radius - LA.norm(np.abs(x) + epsilon, p) ** p + np.dot(weights, np.abs(x))  
        gamma_k_list += [gamma_k]
        
        if gamma_k < 0: 
            Flag_gamma_pos = 'Fail'
            print('The current Gamma is not positive!!!')
            sys.exit(1)
            
        # Solve the subproblem: weighted l1 ball projection, Eq(9) in the paper
        # consider the case when w^Ty <= r, this should return r
        x_proj_nonnegative, lam = simplex_RT.bisection(sign_of_y, lam, y, gamma_k, weights) 
        ################  >??????????????????????? is x_proj_nonnegative ???
        ## use weighted l1 norm projection instead
        # x_proj_nonnegative = projection_onto_weighted_l1_norm_ball(y, weights, radius)
        x_proj_nonnegative = np.abs(x_proj_nonnegative)

        num_nonzeros = np.count_nonzero(x_proj_nonnegative)
        res_nonzero += [num_nonzeros]
        
        # compute the solution with sign and the objective value
        x_current = sign_of_y * x_proj_nonnegative
        objective_value = loss(x_current, y)
        
        # whether the update condition is triggered for epsilon, Eq(13)

        # compute the distance between x_current and x (x_old) (L2 norm)
        local_reduction = x_current - x    # in the limit, it should be zero
        local_reduction_norm = LA.norm(local_reduction, 2)

        # Update epsilon according to Eq(13)
        sign_weight = np.sign(local_reduction) * weights # useless, since we take the norm 
        condition_left = local_reduction_norm * LA.norm(sign_weight, 2) ** Tau
        condition_right = M
        

        # Determine whether to trigger the update condition  
        # The beta function is defined in Eq(25)       
        beta_of_current_x = np.abs(LA.norm(x_proj_nonnegative, p)**p - radius)

        epsilon_seq += [epsilon]
        if condition_left <= condition_right:
            count_trigger += 1
            # theta is defined in the Eq(38.5)
            theta = np.minimum(beta_of_current_x, 1/(np.sqrt(iter)))
            # update epsilon
            epsilon =  theta * epsilon
            
        
        eps_norm = LA.norm(epsilon, np.inf)
                    
        # Checking the termination condition
        # collect the active index from the (k+1)-th solution, x^{k+1}
        active_index_outer = []  
        for ind in range(n):
            if x_proj_nonnegative[ind] > 0:
                active_index_outer.append(ind)
                
        # Determine whether the inactive set remains unchanged

        # Determine whether this collection is empty
        if active_index_outer: # Nonempty active set I(x^k), which defined in the Notation section
            # Our lemma shows the I(x^k) is nonempty

            # Begin to calculate the residual 
            y_active_index_outer = y_bar[active_index_outer]
            x_active_index_outer = x_proj_nonnegative[active_index_outer] # (k+1)th iterate
            weights_active_index_outer = weights[active_index_outer]
        
            # Compute multiplier lambda and two residuals
            # definition of lambda_new is given by Eq(12)
            lambda_new = np.sum(y_active_index_outer - x_active_index_outer) / np.sum(weights_active_index_outer)
            lambda_seq += [lambda_new]

            # alpha residual and beta residual are defined in Eq(25) and Eq(39)
            residual_alpha = (1/n) * np.sum(np.abs((y_bar - x_proj_nonnegative) * x_proj_nonnegative - p * lambda_new * x_proj_nonnegative ** p))
            residual_beta = (1/n) * beta_of_current_x

            res_alpha += [residual_alpha]
            res_beta += [residual_beta]
            res_alpha0 = res_alpha[0]
            res_beta0 = res_beta[0]

            # Step 6 of our algorithm: go to the (k+1)-th iterate.
            x = x_proj_nonnegative          # (k+1)-th solution
            x_iterate += [x]  # store each iterate

            # print('{0:3d}: Obj = {1:4.3e}, alpha = {2:4.3e}, beta = {3:4.3e}, eps = {4:4.3e}, dual = {5:4.3e}, #nonzeros = {6:2d}, x-x_p = {7:4.3e}'.format(iter, objective_value, residual_alpha, residual_beta, eps_norm, lam, num_nonzeros, local_reduction_norm), end=' ')
            # print()
            print_iteration_info(iter, objective_value,residual_alpha, residual_beta, eps_norm, lambda_new, num_nonzeros, local_reduction_norm, gamma_k)
            

            # Check the stopping criteria, defined in Eq(39)
            if np.maximum(residual_alpha, residual_beta) <= tol * np.max([res_alpha[0], res_beta[0], 1]) or iter >= Iter_max:
                if iter >= Iter_max:
                    Flag_gamma_pos = 'Fail'
                    print("The solution is not so good")
                    break
                else:
                    Flag_gamma_pos = 'Success'
                    x_final = sign_of_y * x_proj_nonnegative  # element-wise product
                    break
    return x_final

