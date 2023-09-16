import math
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import KDTree
from spa import SuccessiveProj

N = 1000
K = 3
def runif_in_simplex(n, vertex, sigma=1):
  ''' Return uniformly random vector in the n-simplex '''

  k = np.random.exponential(scale=1.0, size=n)
  w = k/sum(k)
  return np.dot(vertex,np.transpose(w)) + np.random.normal(scale=sigma, size=K-1)

def get_sample(N_sample, vertex):
    samples = np.array([runif_in_simplex(n=K,vertex=vertex) for _ in range(N_sample)]).reshape(-1,K-1)
    return samples

def plot_simplex(vertex,samples):
    vertex_t = np.transpose(vertex)
    plt.scatter(samples[:,0], samples[:,1], s=0.3, label="original sample")
    plt.scatter(vertex_t[:,0], vertex_t[:,1], c="r", label="true vertices")
    #plt.show()


def max_pairwise_distance(arr):
    # Calculate pairwise Euclidean distances
    pairwise_distances = squareform(pdist(arr, 'euclidean'))
    np.fill_diagonal(pairwise_distances, 0)

    # Find the maximum distance
    max_distance = np.max(pairwise_distances)
    
    return max_distance


def knn(samples,M=50,N_n=3):
    l_max = max_pairwise_distance(samples)
    eps = l_max/M
    return eps

def points_within_epsilon(points, epsilon):
    # Create a KD-Tree from the input points
    kdtree = KDTree(points)
    
    # Initialize an empty list to store the results
    results = []
    
    # Query the KD-Tree for points within epsilon for each point
    for j, point in enumerate(points):
        indices = kdtree.query_ball_point(point, epsilon)
        neighbors = [points[i] for i in indices if i!=j]
        if len(neighbors) >= 3:
            results.append(np.mean(neighbors,axis=0))
    
    return np.array(results).reshape(-1,K-1)


if __name__ == '__main__':

    vertex=np.array([[1,2,5],[1,4,2]])
    samples = get_sample(N_sample = N, vertex=vertex)
    eps = knn(samples)
    print(eps)
    X_tilde = points_within_epsilon(samples,eps)
    vertices_knn = SuccessiveProj(X_tilde,K)
    vertices_spa = SuccessiveProj(samples,K)
    plot_simplex(vertex, samples)
    plt.scatter(vertices_knn[:,0], vertices_knn[:,1],c="blue",label="KNN-SPA reconstruction")
    plt.scatter(vertices_spa[:,0], vertices_spa[:,1],c="green",label="SPA reconstruction")
    plt.legend()
    plt.title("Sigma = 1")
    plt.show()
# plot_simplex(vertex=np.array([[1,2,5],[1,4,2]]))
# Example usage:


