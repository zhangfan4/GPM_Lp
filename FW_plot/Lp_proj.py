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
                x = x * signum
            weights = p * (x + epsilon) ** (p-1)
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
            # x_opt, lam = WeightSimplexProjection(n, y, signum, gamma_k, weights)  # x_opt: R^n
            x_opt, lam = simplex_RT.bisection(signum, lam, y, gamma_k, weights)  # x_opt: R^n
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
                epsilon = np.minimum(error_appro, 1/(np.sqrt(count))) * epsilon
                counter = counter + 1
                # print()
                
            eps_norm = LA.norm(epsilon,np.inf)
                        
            #%% Checking the termination conditon
            
            act_ind_outer = []  # collect the active index from the (k+1)-th solution
            for ind in range(len(x_opt)):
                if x_opt[ind] > 0:
                    act_ind_outer.append(ind)
                   
            # Determine the inactive set whether remains unchanged
            #%% Determine whether this collection is empty
            if act_ind_outer: # Nonempty inactive set I. Our lemma shows the I(x^k) is nonempty

            # Begin to calculate the residual 
                y_act_ind_outer = bar_y[act_ind_outer]
                x_act_ind_outer = x_opt[act_ind_outer] #(k+1)th iterate
                weights_ind_outer = weights[act_ind_outer]
            
                ## Check this formula
                #%% Compute multiplier lambda and tow residuals
                lambda_opt = np.divide(np.sum(y_act_ind_outer - x_act_ind_outer), np.sum(weights_ind_outer))
                lambda_opt_seq += [lambda_opt]

                residual_alpha = (1/n) * np.sum(np.abs((bar_y - x_opt) * x_opt - p * lambda_opt * x_opt ** p))
                residual_beta = (1/n) * error_appro

                res_alpha += [residual_alpha]
                res_beta += [residual_beta]
                res_alpha0 = res_alpha[0]
                res_beta0 = res_beta[0]

                # Step 6 of our algorithm: go to the (k+1)-th iterate.
                x = x_opt          # (k+1)-th solution
                res_iterate += [x]  # store each iterate

                print('{0:3d}: Obj = {1:4.3e}, alpha = {2:4.3e}, beta = {3:4.3e}, eps = {4:4.3e}, dual = {5:4.3e}, #nonzeros = {6:2d}, x-x_p = {7:4.3e}'.format(count, obj_k, residual_alpha, residual_beta, eps_norm, lam, num_nonzeros, local_reduction_norm), end=' ')
                print()
                # Check the stopping criteria
                if np.maximum(residual_alpha, residual_beta) <= tol * np.max([res_alpha[0], res_beta[0], 1]) or count >= Iter_max:
                    if count >= Iter_max:
                        Flag_gamma_pos = 'Fail'
                        print("The solution is not so good")
                        break
                    else:
                        Flag_gamma_pos = 'Success'
                        x_final = signum * x_opt  # element-wise product
                        break
    return x_final

#%% Call Newton's method to refine the final solutions
# def func(x, y, theta, p):
#     try:
#         obj = x - y + theta * p * np.power(x, p-1)
#     except RuntimeWarning:
#         print("Error in func!")
#     return obj

# def derivFunc(x, y, theta, p):
#     try:
#         deri = 1 + theta * p * (p-1) * np.power(x, p-2)
#     except RuntimeWarning:
#         print("Error in deriFunc!")
#     return deri

# def root_refine(x_ini, y , p, theta, radius):
#     k = len(x_ini)
#     x_final = np.zeros(k,dtype=np.float64)
#     for i in range(k):
#         optimum = optimize.newton(func, x_ini[i], args=(y[i], theta, p,))
#         x_final[i] = optimum
#     return x_final



# #%% the main function
# if __name__ == '__main__':
#     #%% Input parameters
#     Flag_gamma = []
#     #%%==================================
#     p = 0.5 # A key parameter
#     #====================================
#     power = 1/p
#     # data_dim = 2
#     radius = 1.0
#     # y = point_Projected(radius, data_dim)
#     # y = np.array([0.499838, 0.49832], dtype= np.float64)
#     y = np.array([0.5, 0.5, 0.5], dtype=np.float64)
#     dim = len(y)
#
#     x_ini = np.zeros(dim, dtype=np.float64) # the initial points for the algorithm
#
#
#     #%% Randomly generating epsilon.
#     # epsilon = np.array([0.1,0.1], dtype=np.float64) # with fixed epsilon
#     rand_num = np.random.uniform(0, 1, dim)
#     abs_norm = LA.norm(rand_num, ord=1)
#     epsilon = 0.9 * (rand_num * radius / abs_norm)**power  # ensure that the point is feasible.
#     # epsilon = np.load('epsilon.npy')
#     # np.save('epsilon_ini.npy', epsilon)
#     #%% Proposed reweighted l1-ball projection algorithm.
#     info = '   The variable info during iteration '
#     print('*'*40)
#     print(info)
#     print('*'*40)
#
#     '''
#         Method I: Proposed reweighted l1-ball projection algorithm.
#     '''
#     x_final_weighted = WeightLpBallProjection(dim, x_ini, y, p, radius, epsilon)
#
#     Flag_gamma += [Flag_gamma_pos]
#
#     '''
#         Method II: Algorithm from IEEE 2019
#     '''
#     # x_final_root, theta_opt, Flag_string = root_searching(y, p, radius)
#     #%% We need the following obtained parameter: all of them are with the same length
#     # res_iterate
#     # epsilon_seq
#     # weights_seq
#     # gamma_k_list
#     # slope = np.zeros(count, dtype=np.float64)
#     # bias = np.zeros(count, dtype=np.float64)
#     # #%% Plot the illustration
#     # for i in range(count):
#     #         slope[i] = - np.divide(weights_seq[i][0],weights_seq[i][1])
#     #         bias[i] = np.divide(gamma_k_list[i], weights_seq[i][1])
#
#     #%% Plot results
#
#     # fig = plt.figure(1)
#     # cnt1 = np.arange(len(res_alpha)) # Optimality
#     # cnt2 = np.arange(len(res_beta))  # Feasibility
#
#     # fig1, = plt.plot(cnt1, res_alpha, linewidth=1, label = r'$\alpha(x^k)$')
#     # fig2, = plt.plot(cnt2, res_beta,'--', linewidth=1, label = r'$\beta(x^k)$')
#     # plt.yscale('log')
#     # plt.xlabel(r'Iteration $k$')
#     # plt.ylabel(r'$\alpha(x^k)$ and $\beta(x^k)$ in log-scale')
#     # first_legend = plt.legend(handles = [fig1,fig2], loc='upper right', shadow=True)
#     # plt.show()
#
