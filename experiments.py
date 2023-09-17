from vertexH import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from pyemd import emd
import seaborn as sns

import numpy as np
from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment

# Compute the pairing of points with the minimum Euclidean distance
def compute_min_distance_pairing(point_cloud1, point_cloud2):
    # Calculate the pairwise Euclidean distances between all points in both clouds
    distance_matrix = distance.cdist(point_cloud1, point_cloud2)

    # Solve the assignment problem using the Hungarian algorithm to minimize the total distance
    row_ind, col_ind = linear_sum_assignment(distance_matrix)

    # Return the indices of paired points and the minimum total distance
    paired_points_indices = list(zip(row_ind, col_ind))
    min_total_distance = distance_matrix[row_ind, col_ind].sum()

    return paired_points_indices, min_total_distance


def eps_exp(vertex, samples, eps_range):
    error1 = []
    #error2 = []
    for eps in eps_range:
        X_tilde = compute_points_within_epsilon(samples,eps)
        vertices_knn = SuccessiveProj(X_tilde,K=3)
        #vertices_spa = SuccessiveProj(samples,K=3)
        _, emd1 = compute_min_distance_pairing(np.transpose(vertex), vertices_knn)
        #_, emd2 = compute_min_distance_pairing(np.transpose(vertex), vertices_spa)
        error1.append(emd1)
        #error2.append(emd2)
    plt.plot(eps_range,error1, label = "KNN-SPA")
    #plt.plot(np.linspace(0.1,1,100),error2, label = "SPA")
    plt.xlabel("epsilon")
    plt.ylabel("sum of euclidean distances")
    plt.legend()
    plt.show()


def N_exp(vertex,samples,N_range):
    error1 = []
    #error2 = []
    eps = knn(samples)
    for N in N_range:
        X_tilde = compute_points_within_epsilon(samples,eps,N=N)
        vertices_knn = SuccessiveProj(X_tilde,K=3)
        vertices_spa = SuccessiveProj(samples,K=3)
        _, emd1 = compute_min_distance_pairing(np.transpose(vertex), vertices_knn)
        #_, emd2 = compute_min_distance_pairing(np.transpose(vertex), vertices_spa)
        error1.append(emd1)
        #error2.append(emd2)
    plt.plot(N_range,error1, label = "KNN-SPA")
    #plt.plot(N_range,error2, label = "SPA")
    plt.xlabel("N")
    plt.ylabel("sum of euclidean distances")
    plt.legend()
    plt.show()


def N_eps_exp(samples,eps,N):
    X_tilde = compute_points_within_epsilon(samples,eps,N)
    vertices_knn = SuccessiveProj(X_tilde,K=3)
    #vertices_spa = SuccessiveProj(samples,K=3)
    _, emd1 = compute_min_distance_pairing(np.transpose(vertex), vertices_knn)
    #_, emd2 = compute_min_distance_pairing(np.transpose(vertex), vertices_spa)
    return emd1
    #error2.append(emd2)
    


if __name__ == '__main__':
    vertex = np.array([[1,2,5],[1,4,2]])
    N_range = np.arange(1,20)
    samples = np.load("samples_bigvar.npy")
    N_exp(vertex,samples,N_range)
#     eps_range = np.linspace(0.1,2,50)
#     N_range = np.arange(1,10)
#     error = []
#     for eps in eps_range:
#         for N in N_range:
#             Z = N_eps_exp(samples,eps, N)
#             error.append(Z)
#     # Create a contour plot
#     plt.figure(figsize=(8, 6))
#     combi = np.array(np.meshgrid(eps_range,N_range)).T.reshape(-1, 2)
#     values = np.concatenate((np.array(error).reshape(-1,1),combi),axis=1)
#     x,y,z = values[:,0],values[:,1],values[:,2]
#     sns.heatmap(values,cmap="crest")
#     plt.show()

# # # Show the plot
# # plt.show()
# #     samples = np.load("samples_bigvar.npy")
# #     eps_exp(vertex,samples,np.linspace(0.1,3,100))
    