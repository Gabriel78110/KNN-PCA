import numpy as np

def SuccessiveProj(R, K):
    n = R.shape[0]
    
    if R.ndim == 1:
        r = 1
        vertices = np.zeros((K,))
        Y = np.column_stack((np.ones(n), R))
        
        for i in range(K):
            ind = np.argmax(np.sum(Y**2, axis=1))
            vertices[i] = R[ind]
            u = Y[ind, :] / np.sqrt(np.sum(Y[ind, :]**2))
            length = np.dot(Y, u)
            Y -= np.outer(length, u)
        
        return vertices
    
    r = R.shape[1]
    vertices = np.zeros((K, r))
    Y = np.column_stack((np.ones(n), R))
    
    for i in range(K):
        ind = np.argmax(np.sum(Y**2, axis=1))
        vertices[i, :] = R[ind, :]
        u = Y[ind, :] / np.sqrt(np.sum(Y[ind, :]**2))
        length = np.dot(Y, u)
        Y -= np.outer(length, u)
    
    return vertices
