import math
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import KDTree
from spa import SuccessiveProj

N = 1000
K = 3
def runif_in_simplex(n, vertex, sigma=0.5):
  ''' Return uniformly random vector in the n-simplex '''

  k = np.random.exponential(scale=1.0, size=n)
  w = k/sum(k)
  return np.dot(vertex,np.transpose(w)) + np.random.normal(scale=sigma, size=K-1)

def get_sample(N_sample, vertex):
    samples = np.array([runif_in_simplex(n=K,vertex=vertex) for _ in range(N_sample)]).reshape(-1,K-1)
    return samples

def plot_simplex(vertex,samples):
    vertex_t = np.transpose(vertex)
    vertex_t = np.concatenate((vertex_t,vertex_t[0].reshape(1,-1)),axis=0)
    plt.scatter(samples[:,0], samples[:,1], s=0.3, label="original sample")
    plt.plot(vertex_t[:,0], vertex_t[:,1], "r--", marker='o', label="true vertices")
    #plt.show()


def max_pairwise_distance(arr):
    # Calculate pairwise Euclidean distances
    pairwise_distances = squareform(pdist(arr, 'euclidean'))
    np.fill_diagonal(pairwise_distances, 0)

    # Find the maximum distance
    max_distance = np.max(pairwise_distances)
    
    return max_distance


def knn(samples,M=20):
    l_max = max_pairwise_distance(samples)
    eps = l_max/M
    return eps

# def points_within_epsilon(points, epsilon,N=3):
#     # Create a KD-Tree from the input points
#     kdtree = KDTree(points)
    
#     # Initialize an empty list to store the results
#     results = []
    
#     # Query the KD-Tree for points within epsilon for each point
#     for j, point in enumerate(points):
#         #print("epsilon=",epsilon)
#         indices = kdtree.query_ball_point(point, epsilon)
#         neighbors = [points[i] for i in indices if i!=j]
#         if len(neighbors) >= N:
#             results.append(np.mean(neighbors,axis=0))
    
#     return np.array(results).reshape(-1,K-1)



def compute_points_within_epsilon(point_cloud, epsilon, N=3):
    kdtree = KDTree(point_cloud)
    result = []

    for point in point_cloud:
        # Query the KD-Tree to find points within epsilon distance
        indices_within_epsilon = kdtree.query_ball_point(point, epsilon)

        if len(indices_within_epsilon) >= N:
            # If more than N points within epsilon distance, keep only the closest N points
            distances = kdtree.query(point, k=N+1)  # Include the point itself in the query
            closest_indices = distances[1]
            result.append(np.mean(point_cloud[closest_indices],axis=0))

    return np.array(result).reshape(-1,K-1)

if __name__ == '__main__':

   

    vertex=np.array([[1,2,5],[1,4,2]])
    #samples = get_sample(N_sample = N, vertex=vertex)
    samples = np.load("samples_smallvar.npy")
    #np.save('my_array.npy', samples)
    eps = knn(samples)
    print(eps)
    #X_tilde = points_within_epsilon(samples,eps)
    X_tilde = compute_points_within_epsilon(samples,eps)
    vertices_knn = SuccessiveProj(X_tilde,K)
    vertices_knn = np.concatenate((vertices_knn, vertices_knn[0].reshape(1,-1)),axis=0)
    vertices_spa = SuccessiveProj(samples,K)
    vertices_spa = np.concatenate((vertices_spa, vertices_spa[0].reshape(1,-1)),axis=0)
    plot_simplex(vertex, samples)
    plt.plot(vertices_knn[:,0], vertices_knn[:,1],"b--", marker='o',label="KNN-SPA reconstruction")
    plt.plot(vertices_spa[:,0], vertices_spa[:,1],"g--", marker='o',label="SPA reconstruction")
    plt.legend()
    plt.title("Sigma = 0.5")
    plt.show()


