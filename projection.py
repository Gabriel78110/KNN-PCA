import numpy as np
from numpy.linalg import svd


def get_projected_points(samples,K=3):
    '''Input: samples = d * N_samples array
    Output: d * N_samples array where i^th column is the i^th projected point'''
    row_means = np.mean(samples, axis=1)
    means_expanded = np.outer(row_means, np.ones(samples.shape[1]))
    X_bar = samples - means_expanded    

    """SVD STEP"""
    U, _, _ = svd(X_bar)
    U_s = U[:,:K-1]
    H = np.dot(U_s,np.transpose(U_s))
    return means_expanded + np.dot(H,X_bar)
