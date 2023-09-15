import math
import random
import numpy as np
import matplotlib.pyplot as plt

N = 1000
K = 3
def runif_in_simplex(n, vertex, sigma=0.1):
  ''' Return uniformly random vector in the n-simplex '''

  k = np.random.exponential(scale=1.0, size=n)
  w = k/sum(k)
  return np.dot(vertex,np.transpose(w)) + np.random.normal(scale=sigma, size=K-1)

def get_sample(N_sample, vertex):
    samples = np.array([runif_in_simplex(n=K,vertex=vertex) for _ in range(N_sample)]).reshape(-1,K-1)
    return samples

def plot_simplex(vertex):
    vertex_t = np.transpose(vertex)
    samples = get_sample(N_sample = N, vertex=vertex)
    plt.scatter(samples[:,0], samples[:,1], s=0.2)
    plt.scatter(vertex_t[:,0], vertex_t[:,1], c="r")
    plt.show()


plot_simplex(vertex=np.array([[1,2,5],[1,4,2]]))
