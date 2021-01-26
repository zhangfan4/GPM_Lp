"""
    auxiliary projections
    includes:
        1. projection onto a hyperplane H(w, b) = {x| w^Tx = b}. Given y
            min 0.5 * || x - y||_2^2
            s.t. x \in H(w, b)
        2. projection onto a l1 norm ball with radius r, B =  {x| ||x||_{w,1}\leq r}
            min 0.5 * || x - y||_2^2
            s.t. x \in B

            where ||x||_{w,1} = sum_{i} w_i|x|_i = sum_{i} w_i|x_i|
"""

from numpy import linalg as LA
import numpy as np



def projection_onto_hyperplane(y, w, b):
    """
    projection y onto a hyperplane defined by w and b
    formally, the solution to the following problem
    min 0.5 * ||x - y||_2^2
    s.t. w^Tx = b,

    the solution of the above problem is given by 
    x_proj = y - (w^Ty - b)/(||w||_2^2) * w


    :parameter w, b: the parameter of a hyperplane H(w, b)
        w is a non-zero vector of same size as x, b is a scalar
        H(w, b) = {x| w^Tx = b}
    :parameter y: the  given point (a vector of same size as x) to be projected

    :returns: 
        coef: the coeffficient of w in the solution 
        x_proj: the projection of y onto the hyperplane H(w, b)
            x_proj = y - coef * w
            
    """
    dominator = w.dot(y) - b
    dedominator = LA.norm(w, 2) ** 2
    try:
        coef = dominator / dedominator
    except ZeroDivisionError:
        print("Error! - derivation zero for scalar2 =", scalar2)
        sys.exit(1)

    x_proj = y - coef * w
    return x_proj, coef


def projection_onto_weighted_l1_norm_ball(y, weights, radius):
    """
    projection onto a weighted L1-ball
    formally, 

    min 0.5 * ||x - y||_2^2
    s.t. weights^T |x| = radius,

    where |x|_i = |x_i|

    :parameter radius: the radius of the weighted L1-ball
    :parameter weights: the weight vector
    :parameter y: the point to be projected

    :return: x_proj: projection of y onto the weighted L1 ball 
    """

    # make the point projected non-negative
    y_sign = np.sign(y)
    y_proj = y_sign * y

    # we run the algorithm until the active index doesn't change after the update 
    active_index = range(len(y))

    # TODO: SEE if we can accelerated this projection!
    while True:
        # we only update entry that is positive
        y_proj_active = y_proj[active_index]
        weights_active = weights[active_index]

        # projection onto the surface of the weighted l1 norm ball
        x_proj_onto_hyperplane, coef = projection_onto_hyperplane(y_proj_active, weights_active, radius)

        # throw away the entries that is negative
        y_proj_active = np.maximum(x_proj_onto_hyperplane, 0.0)
        y_proj[active_index] = y_proj_active

        # update the active set
        active_index = (y_proj > 0)

        # if x_proj_onto_hyperplane(i) < 0, we need to project again
        inactive_index_cardinality = sum(x_proj_onto_hyperplane < 0)
        if inactive_index_cardinality == 0:
            x_opt = y_proj * y_sign
            break

    return x_opt



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