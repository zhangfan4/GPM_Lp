"""
    Find the largest eigenvalue of a matrix A via power iteration method. 
    see 
    https://en.wikipedia.org/wiki/Power_iteration
    for details
"""

import numpy as np
from numpy import linalg as LA

def power_iteration(A, num_simulations):
    """
    Power iteration Algorithm for largest eigenvalue
    :param A: a diagonalizable matrix
    :param num_simulations: max iteration number
    :return: eigenvector b_k
    """
    b_k = np.random.rand(A.shape[1])

    for _ in range(num_simulations):
        # calculate the matrix-by-vector product Ab
        b_k1 = A.dot(b_k)

        # calculate the norm
        b_k1_norm = LA.norm(b_k1)

        # re normalize the vector
        b_k = b_k1 / b_k1_norm

    return b_k


def compute_largest_eigenvalue(A, num_simulations):
    """
    use the approximated eigenvector returned by Power Iteration method to compute
    the largest eigenvalue of the matrix A
    :param A: a matrix of size (m, n)
    :param b_k: a approximated eigenvector of A with size (n, 1) returned by
        power_iteration(A, , num_simulations)
    :return max_eigenvalue: the maximum absolute eigenvalue of the matrix A
    """
    b_k = power_iteration(A, num_simulations)
    return b_k.dot(A).dot(b_k) / (b_k.dot(b_k))