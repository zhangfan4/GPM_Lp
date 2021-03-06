#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 15:20:52 2021

@author: moumatsu
"""

import numpy as np


def bisection(weight, y, radius, lamb):
    """
    input:
        y: a vector in the positive orthant
    
    output:
        root for the objective function
    """
    tolerance = 1e-15   # precision for bisection method
    low = 0

    act_ind = np.where(abs(weight) > 1e-15)[0]
    high = max(y[act_ind] / weight[act_ind])
    value_of_high = np.sum(weight.dot(np.maximum(y - high * weight, 0))) - radius
    value_of_low = np.sum(weight.dot(np.maximum(y, 0))) - radius

    assert value_of_high * value_of_low < 0, "The sign must not be the same"
    
    while True:
        value_of_lamb = np.sum(weight.dot(np.maximum(y - lamb * weight, 0))) - radius
        
        
        
        if abs(value_of_lamb) <= tolerance:
            break
        
        # bisection method
        if value_of_lamb < 0:
            high = lamb
        else:
            low = lamb
        
        
        if abs(high - low) <= tolerance:
            print(abs(value_of_lamb))
            print("Fail to find the root")
            break
        
        lamb = (high + low) / 2
        
    
    x_opt = np.maximum(y - lamb * weight, 0)
    
    return x_opt, lamb