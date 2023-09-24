import math
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment
from scipy.spatial import KDTree,cKDTree
from spa import SuccessiveProj
from utils import find_max_distance_to_average, find_min_distance_to_average

N_sample = 1000
K = 3
d = 2

def runif_in_simplex(n, vertex, sigma=0.5,d=2):
  ''' Return uniformly random vector in the n-simplex 
  vertex: d * K array'''

  k = np.random.exponential(scale=1.0, size=n)
  w = k/sum(k)
  #+ np.random.normal(scale=sigma, size=d)
  return np.dot(vertex,np.transpose(w))

def get_sample(N_sample, vertex,pure_nodes=10,sigma=0.5,d=2):
    #+ np.random.normal(scale=sigma, size=(K*pure_nodes,d))
    vertices_pure = np.tile(np.transpose(vertex),(pure_nodes,1))
    samples = np.concatenate((vertices_pure,np.array([runif_in_simplex(n=K,vertex=vertex,d=d) for _ in range(N_sample-pure_nodes*K)]).reshape(-1,d)))
    return samples

def plot_simplex(vertex,samples,ax,plot_data=True):
    vertex_t = np.transpose(vertex)
    vertex_t = np.concatenate((vertex_t,vertex_t[0].reshape(1,-1)),axis=0)
    if plot_data:
        ax.scatter(samples[:,0], samples[:,1], s=0.3, label="sample data",c="black")
        ax.plot(vertex_t[:,0], vertex_t[:,1], "k--", marker='o', label="ideal simplex")
    else:
        ax.plot(vertex_t[:,0], vertex_t[:,1], "k--", marker='o', label="ideal simplex")
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


def compute_points_within_epsilon(point_cloud, epsilon, N,t,d=2):
    """point_cloud: N_samples * d numpy array"""

    # Create a KD-tree for fast neighbor searches
    kdtree = cKDTree(point_cloud)

    # Query for all points within epsilon distance
    neighbor_indices = kdtree.query_ball_tree(kdtree, r=epsilon)

    # Initialize a list to store the filtered and replaced points
    filtered_points = []

    for i, point in enumerate(point_cloud):
        #neighbor_indices[i].remove(i)
        
        # If there are fewer than t neighbors, discard this point
        if len(neighbor_indices[i]) < t:
            continue

        # If there are more than t neighbors, but fewer than N neighbors, 
        # take the average of all neighbors.
        if len(neighbor_indices[i]) <= N:
            avg_neighbor = np.mean([point_cloud[j] for j in neighbor_indices[i]], axis=0)
        else:
            # Get the N closest neighbors
            distances = kdtree.query(point, k=N)
            avg_neighbor = np.mean([point_cloud[j] for j in distances[1]], axis=0)

        # Append the average neighbor to the filtered list
        filtered_points.append(avg_neighbor)

    return np.array(filtered_points).reshape(-1,d)

# Compute the pairing of points with the minimum Euclidean distance
def compute_min_distance_pairing(point_cloud1, point_cloud2,max=True):
    # Calculate the pairwise Euclidean distances between all points in both clouds
    distance_matrix = distance.cdist(point_cloud1, point_cloud2)
    row_ind, col_ind = linear_sum_assignment(distance_matrix)


    # Return the indices of paired points and the minimum total distance
    paired_points_indices = list(zip(row_ind, col_ind))
    if not max:
        min_total_distance = distance_matrix[row_ind, col_ind].sum()

        return paired_points_indices, min_total_distance

    min_total_distance = np.max(distance_matrix[row_ind, col_ind])
    return paired_points_indices, min_total_distance



if __name__ == '__main__':

   

    vertex=np.array([[1,2,5],[1,4,2]])
    #samples = get_sample(N_sample = N, vertex=vertex)
    samples = np.load("samples_bigvar.npy")
    #np.save('my_array.npy', samples)
    eps = knn(samples)
    #X_tilde = points_within_epsilon(samples,eps)
    X_tilde = compute_points_within_epsilon(samples,epsilon=0.1,N=4,t=3)
    vertices_knn = SuccessiveProj(X_tilde,K)
    vertices_knn = np.concatenate((vertices_knn, vertices_knn[0].reshape(1,-1)),axis=0)
    vertices_spa = SuccessiveProj(samples,K)
    vertices_spa = np.concatenate((vertices_spa, vertices_spa[0].reshape(1,-1)),axis=0)


    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,3))
    plot_simplex(vertex, samples,ax1)
    ax1.plot(vertices_spa[:,0], vertices_spa[:,1],"g--", marker='o',label="SPA")

    plot_simplex(vertex, samples,ax2,plot_data=False)
    ax2.scatter(X_tilde[:,0], X_tilde[:,1], s=0.3, label="psuedo-points",c="b")
    ax2.plot(vertices_knn[:,0], vertices_knn[:,1],"c--", marker='o',label="PPSPA (step 2)")
    ax1.legend()
    ax2.legend()
    #plt.title(f"t = {t}")
    plt.show()