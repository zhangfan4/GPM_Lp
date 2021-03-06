#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 19:28:15 2021

@author: moumatsu
"""

import time
import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

# customed module
import objective as obj
import projection
import computation    # used for compute maximum eigenvalue

def logger(func):
    def wrapper(*args, **kwargs):
        print("%s is running" % func.__name__)
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        time_cost = end - start
        result['time'] = time_cost
        print("function run time is %.2f" % time_cost)
        return result
    return wrapper
        

@logger
def run_proj_lpball(A, x0, b, p, radius, MAX_ITERS=1e3):
    assert (LA.norm(x0, p)**p <= radius), "x0 must lie in the lp norm ball"
    n,  = x0.shape
    # machine precision
    precision = 1e-6
    
    # tolerance
    tolerance = 1e-6
    
    nonzero_num = n
    
    # record the information
    result = {}
    result['obj_val'] = []
    result['nonzero'] = []
    result['residual'] = []
    
    L = computation.max_eigenvalue(A.T.dot(A))
    step_size = 1.0 / L
    lamb = 0
    
    t = 1
    
    #  run algorithm
    for out_iter in range(1, int(MAX_ITERS)+1):
        # step_size = 1.0
        grad = obj.gradient(A, x0, b)
        
        # gradient step
        z = x0 - step_size * grad   
        
        if LA.norm(z, p)**p <= radius:
            print("z is feasible")
            x_new = z
        
        elif abs(LA.norm(x0, p)**p - radius) <= precision:
            # x0 is on the boundary
            
            # projection onto the weighted l1 norm ball
            print("projection onto the weighted l1 norm ball")
            act_ind = np.where(abs(x0) > precision)[0]
            z[[x for x in range(n) if x not in act_ind]] = 0
            weight = np.zeros(n)
            weight[act_ind] = p * abs(x0[act_ind]) ** (p - 1)
            radius_L1 = weight[act_ind].dot(abs(x0[act_ind]))
            x_new, lamb = projection.proj_weightedl1ball(np.abs(z), 
                                                         weight, 
                                                         radius_L1, 
                                                         lamb,
                                                         'bisection')
            x_new *= np.sign(z)
            
            if LA.norm(x_new - x0) <= tolerance:
                break
            
        else:
            # x0 is inside the Lp norm ball
            
            # projecyion onto the lp norm ball
            print("projection onto the lp norm ball")
            if LA.norm(grad) <= tolerance:
                break
            
            x_new, nonzero_num = projection.proj_lpball(z, p, x0, radius)
            
        nonzero_num = np.count_nonzero(x_new)
        residual = LA.norm(x0 - x_new)
        
        # update the current point
        # x0 = x_new
        t_new = (1 + np.sqrt(1 + 4 * t**2)) / 2
        act_ind = np.where(abs(x_new) > precision)[0]
        x0[[x for x in range(n) if x not in act_ind]] = 0
        x0 = x_new + ((t - 1) / t_new) * (x_new - x0)
        
        # record the iteration information
        obj_val = obj.value(A, x0, b)
        result['obj_val'].append(obj_val)
        result['nonzero'].append(nonzero_num)
        result['residual'].append(residual)
        result['x_opt'] = x_new
        result['# iter'] = out_iter
        
        
        print("s=%3d, error=%4.3e, obj_value=%4.3e, nonzero=%3d" 
              % (out_iter, residual, obj_val, nonzero_num))
        
    if out_iter >= MAX_ITERS:
        print("Not a good solution")
    
    return result



@logger
def run_proj_l1ball(A, x0, b, radius, MAX_ITERS=1e3):
    n,  = x0.shape
    tolerance = 1e-6
    L = computation.max_eigenvalue(A.T.dot(A))
    step_size = 1.0 / L
    
    # record the information
    result = {}
    result['obj_val'] = []
    result['nonzero'] = []
    result['residual'] = []
    
    for out_iter in range(1, int(MAX_ITERS)+1):
        grad = obj.gradient(A, x0, b)
        
        # gradient step
        z = x0 - step_size * grad
        
        # projection onto l1 ball
        x_new = projection.proj_l1ball(z, radius)
            
        nonzero_num = np.count_nonzero(x_new)
        residual = LA.norm(x0 - x_new)
        obj_val = obj.value(A, x0, b)
        
        result['obj_val'].append(obj_val)
        result['nonzero'].append(nonzero_num)
        result['residual'].append(residual)
        result['x_opt'] = x_new
        result['# iter'] = out_iter
        
        print("s=%3d, error=%4.2e, obj_value=%4.3e, nonzero=%3d" 
              % (out_iter, residual, obj.value(A, x0, b), nonzero_num))
        if residual <= tolerance or LA.norm(grad) <= tolerance:
                break
        
        # dupdate current point
        x0 = x_new
        
    return result

