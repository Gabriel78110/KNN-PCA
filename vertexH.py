import math
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import KDTree,cKDTree
from spa import SuccessiveProj

N_sample = 1000
K = 3
d = 2

def runif_in_simplex(n, vertex, sigma=0.5):
  ''' Return uniformly random vector in the n-simplex 
  vertex: d * K array'''

  k = np.random.exponential(scale=1.0, size=n)
  w = k/sum(k)
  return np.dot(vertex,np.transpose(w)) + np.random.normal(scale=sigma, size=d)

def get_sample(N_sample, vertex):
    samples = np.array([runif_in_simplex(n=K,vertex=vertex) for _ in range(N_sample)]).reshape(-1,d)
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


def compute_points_within_epsilon(point_cloud, epsilon, N,t):
    # Create a KD-tree for fast neighbor searches
    kdtree = cKDTree(point_cloud)

    # Query for all points within epsilon distance
    neighbor_indices = kdtree.query_ball_tree(kdtree, r=epsilon)

    # Initialize a list to store the filtered and replaced points
    filtered_points = []

    for i, point in enumerate(point_cloud):
        # Exclude the point itself from the list of neighbors
        neighbor_indices[i].remove(i)
        
        # If there are fewer than t neighbors, discard this point
        if len(neighbor_indices[i]) < t:
            continue

        # If there are more than t neighbors, but fewer than N neighbors, 
        # take the average of all neighbors.
        if len(neighbor_indices[i]) <= N:
            avg_neighbor = np.mean([point_cloud[j] for j in neighbor_indices[i]], axis=0)
        else:
            # Get the N closest neighbors
            distances = kdtree.query(point, k=N+1)
            avg_neighbor = np.mean([point_cloud[j] for j in distances[1]], axis=0)

        # Append the average neighbor to the filtered list
        filtered_points.append(avg_neighbor)

    return np.array(filtered_points).reshape(-1,d)



if __name__ == '__main__':

   

    vertex=np.array([[1,2,5],[1,4,2]])
    #samples = get_sample(N_sample = N, vertex=vertex)
    samples = np.load("samples_bigvar.npy")
    #np.save('my_array.npy', samples)
    eps = knn(samples)
    #X_tilde = points_within_epsilon(samples,eps)
    for t in range(1,20):
        X_tilde = compute_points_within_epsilon(samples,eps,N=3,t=t)
        vertices_knn = SuccessiveProj(X_tilde,K)
        vertices_knn = np.concatenate((vertices_knn, vertices_knn[0].reshape(1,-1)),axis=0)
        vertices_spa = SuccessiveProj(samples,K)
        vertices_spa = np.concatenate((vertices_spa, vertices_spa[0].reshape(1,-1)),axis=0)
        plot_simplex(vertex, samples)
        plt.plot(vertices_knn[:,0], vertices_knn[:,1],"b--", marker='o',label="KNN-SPA reconstruction")
        plt.plot(vertices_spa[:,0], vertices_spa[:,1],"g--", marker='o',label="SPA reconstruction")
        plt.legend()
        plt.title(f"t = {t}")
        plt.show()


