#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 16:43:29 2021

@author: moumatsu
"""

import numpy as np
import numpy.linalg as LA

from run_PGM import run_proj_lpball, run_proj_l1ball
from data_loader import load_data

import objective as obj
from prettytable import PrettyTable



# %% configuration
# m = [1000, 2000, 3000, 4000, 5000]
# n = [1000, 2000, 3000, 4000, 5000]
m=[10000]
n=[10000]

table = PrettyTable()
table.field_names = ['size', 'method', 
                     '#iter', 'time', 
                     'rel_error', 'obj', 
                     '#non-zero']

sizes = zip(m, n)
num_simulations = 1





# %% run algorithm
for size in sizes:
    m, n = size
    
    result = {}
    result['iters_l1'] = 0
    result['iters_lp'] = 0
    result['time_l1'] = 0
    result['time_lp'] = 0
    result['obj_val_l1'] = 0
    result['obj_val_lp'] = 0
    result['nonzero_l1'] = 0
    result['nonzero_lp'] = 0
    result['res_l1'] = 0
    result['res_lp'] = 0

    for _ in range(num_simulations):
        # data generation
        A, x_exact, b = load_data(m, n, s=200)
        
        p = 0.5     # p-norm is used
        radius_lp = 0.8 * LA.norm(x_exact, p)**p     # radius of the lp norm ball
        radius_l1 = 0.8 * LA.norm(x_exact, 1)
        
        # initial point for our algorithm
        x0 = np.zeros(n)
        
        
        #  run PGM method with projection onto the Lp norm ball
        result_lp = run_proj_lpball(A, x0, b, p, radius_lp, 10000)
        
        
        #  run PGM method with projection onto the L1 norm ball
        result_l1 = run_proj_l1ball(A, x0, b, radius_l1, 10000)
        
        
        #  analysis
        x_opt_lp = result_lp['x_opt']
        result['iters_lp'] += result_lp['# iter']
        result['time_lp'] += result_lp['time']
        result['res_lp'] += LA.norm(x_exact-x_opt_lp) / LA.norm(x_exact)
        result['obj_val_lp'] += obj.value(A, x_opt_lp, b)
        result['nonzero_lp'] += np.count_nonzero(x_opt_lp)
        
        x_opt_l1 = result_l1['x_opt']
        result['iters_l1'] += result_l1['# iter']
        result['time_l1'] += result_l1['time']
        result['res_l1'] += LA.norm(x_exact-x_opt_l1) / LA.norm(x_exact)
        result['obj_val_l1'] += obj.value(A, x_opt_l1, b)
        result['nonzero_l1'] += np.count_nonzero(x_opt_l1)
        
    
    for key in result.keys():
        result[key] /= num_simulations
    
    
    table.add_row([n, 'l1', 
                   result['iters_l1'], 
                   result['time_l1'], 
                   result['res_l1'], 
                   result['obj_val_l1'], 
                   result['nonzero_l1']])
    table.add_row([n, 'lp', 
                   result['iters_lp'], 
                   result['time_lp'], 
                   result['res_lp'], 
                   result['obj_val_lp'], 
                   result['nonzero_lp']])


# %% print the result
print(table)

# %% plot the objective value
import matplotlib.pyplot as plt

obj_val_seq_l1 = result_l1['obj_val']
obj_val_seq_lp = result_lp['obj_val']
nonzero_seq_l1 = result_l1['nonzero']
nonzero_seq_lp = result_lp['nonzero']


plt.subplot(1, 2, 1)

plt.plot(obj_val_seq_l1, 'b', label='l1 ball')
plt.plot(obj_val_seq_lp, 'r', label='lp ball')
plt.yscale('log')
plt.legend()
plt.title('objective value')
plt.xlabel('#iteration')

plt.subplot(1, 2, 2)
plt.plot(nonzero_seq_l1, 'b', label='l1 ball')
plt.plot(nonzero_seq_lp, 'r', label='lp ball')
plt.legend()
plt.title('non-zero')
plt.xlabel('#iteration')
plt.show()

    
    
    
    
