#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 14:37:15 2021

@author: moumatsu
"""

import numpy as np


def load_data(m, n, s):
    """
    :param m: number of samples
    :param n: number of features
    :param s: number of zero components
    :return: 
    """
    A = np.random.randn(m, n)
    x_opt = np.random.rand(n)
    # ind = np.random.choice(n, s)
    # x_opt[ind] = 0
    b = A.dot(x_opt)
    
    return A, x_opt, b