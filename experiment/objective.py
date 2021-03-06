#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 20:01:29 2021

@author: moumatsu

Information of the objective function
"""

import numpy as np
import numpy.linalg as LA

def value(A, x, b):
    return 0.5 * LA.norm(A.dot(x) - b) ** 2



def gradient(A, x, b):
    return np.dot(A.T, np.dot(A, x) - b)

