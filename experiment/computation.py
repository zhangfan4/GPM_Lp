#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 19:53:41 2021

@author: moumatsu
"""


import numpy as np

def power_iteration(A, num_simulations=100):
    # Ideally choose a random vector
    # To decrease the chance that our vector
    # Is orthogonal to the eigenvector
    b_k = np.random.rand(A.shape[1])

    for _ in range(num_simulations):
        # calculate the matrix-by-vector product Ab
        b_k1 = np.dot(A, b_k)

        # calculate the norm
        b_k1_norm = np.linalg.norm(b_k1)

        # re normalize the vector
        b_k = b_k1 / b_k1_norm

    return b_k


def max_eigenvalue(A, num_simulations=100):
    b_k = power_iteration(A, num_simulations)
    return b_k.T.dot(A).dot(b_k) / b_k.T.dot(b_k)
