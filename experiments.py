from vertexH import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from pyemd import emd
import seaborn as sns
from projection import get_projected_points

import numpy as np


"---------------------------- SIMULATION DIMENSION ------------------------------------"



def d_simulation(vertex,D=10,S=1,N_sample=1000):
    error_knn, error_spa, error_proj = [], [], []
    K = vertex.shape[1]

    for d in np.arange(2,D):
        e_knn, e_spa, e_proj = [], [], []
        for _ in range(5):
            A = np.random.randint(2,size=(d,2))
            vertex_d = np.dot(A,vertex)
            samples = np.dot(A,np.transpose(get_sample(N_sample = N_sample, vertex=vertex)))
            samples +=  np.random.normal(scale=S, size=(d,N_sample))

            samples_ = get_projected_points(samples)
            eps_proj = find_max_distance_to_average(np.transpose(samples_))/5
            eps_knn = find_max_distance_to_average(np.transpose(samples))/5
            X_proj = compute_points_within_epsilon(np.transpose(samples_),epsilon=eps_proj,N=4,t=3,d=d)
            X_knn = compute_points_within_epsilon(np.transpose(samples),epsilon=eps_knn,N=4,t=3,d=d)
            vertices_proj = SuccessiveProj(X_proj,K)
        
            
            vertices_knn = SuccessiveProj(X_knn,K)
            vertices_spa = SuccessiveProj(np.transpose(samples),K)
            _, emd_knn = compute_min_distance_pairing(np.transpose(vertex_d), vertices_knn)
            _, emd_spa = compute_min_distance_pairing(np.transpose(vertex_d), vertices_spa)
            _, emd_proj = compute_min_distance_pairing(np.transpose(vertex_d), vertices_proj)

            e_knn.append(emd_knn)
            e_proj.append(emd_proj)
            e_spa.append(emd_spa)

        error_knn.append(np.mean(e_knn))
        error_spa.append(np.mean(e_spa))
        error_proj.append(np.mean(e_proj))


    plt.plot(np.arange(2,D),error_knn,"-o",label = "KNN-SPA")
    plt.plot(np.arange(2,D),error_spa,"-o",label = "SPA")
    plt.plot(np.arange(2,D),error_proj,"-o",label = "P-KNN-SPA")
    plt.legend()
    plt.xlabel("dimension (d)")
    plt.ylabel("reconstruction error")
    plt.grid()
    plt.show()

"---------------------------- SIMULATION NOISE ------------------------------------"



def sigma_simulation(vertex,S_range,d=4,N_sample=1000):
    error_knn, error_spa, error_proj = [], [], []
    K = vertex.shape[1]

    for S in S_range:
        e_knn, e_spa, e_proj = [], [], []
        for _ in range(5):
            A = np.random.randint(2,size=(d,2))
            vertex_d = np.dot(A,vertex)
            samples = np.dot(A,np.transpose(get_sample(N_sample = N_sample, vertex=vertex)))
            samples +=  np.random.normal(scale=S, size=(d,N_sample))

            samples_ = get_projected_points(samples)
            eps_proj = find_max_distance_to_average(np.transpose(samples_))/5
            eps_knn = find_max_distance_to_average(np.transpose(samples))/5
            X_proj = compute_points_within_epsilon(np.transpose(samples_),epsilon=eps_proj,N=4,t=3,d=d)
            X_knn = compute_points_within_epsilon(np.transpose(samples),epsilon=eps_knn,N=4,t=3,d=d)
            vertices_proj = SuccessiveProj(X_proj,K)
        
            
            vertices_knn = SuccessiveProj(X_knn,K)
            vertices_spa = SuccessiveProj(np.transpose(samples),K)
            _, emd_knn = compute_min_distance_pairing(np.transpose(vertex_d), vertices_knn)
            _, emd_spa = compute_min_distance_pairing(np.transpose(vertex_d), vertices_spa)
            _, emd_proj = compute_min_distance_pairing(np.transpose(vertex_d), vertices_proj)

            e_knn.append(emd_knn)
            e_proj.append(emd_proj)
            e_spa.append(emd_spa)

        error_knn.append(np.mean(e_knn))
        error_spa.append(np.mean(e_spa))
        error_proj.append(np.mean(e_proj))


    plt.plot(S_range,error_knn,"-o",label = "KNN-SPA")
    plt.plot(S_range,error_spa,"-o",label = "SPA")
    plt.plot(S_range,error_proj,"-o",label = "P-KNN-SPA")
    plt.legend()
    plt.xlabel("Noise level (sigma)")
    plt.ylabel("reconstruction error")
    plt.grid()
    plt.show()

"---------------------------- SIMULATION SAMPLE SIZE ------------------------------------"

def N_simulation(vertex,N_range,d=4,S=1):
    error_knn, error_spa, error_proj = [], [], []
    K = vertex.shape[1]

    for N_ in N_range:
        e_knn, e_spa, e_proj = [], [], []
        for _ in range(5):
            A = np.random.randint(2,size=(d,2))
            vertex_d = np.dot(A,vertex)
            samples = np.dot(A,np.transpose(get_sample(N_sample = N_, vertex=vertex)))
            samples +=  np.random.normal(scale=S, size=(d,N_))

            samples_ = get_projected_points(samples)
            eps_proj = find_max_distance_to_average(np.transpose(samples_))/5
            eps_knn = find_max_distance_to_average(np.transpose(samples))/5
            X_proj = compute_points_within_epsilon(np.transpose(samples_),epsilon=eps_proj,N=4,t=3,d=d)
            X_knn = compute_points_within_epsilon(np.transpose(samples),epsilon=eps_knn,N=4,t=3,d=d)
            vertices_proj = SuccessiveProj(X_proj,K)
        
            
            vertices_knn = SuccessiveProj(X_knn,K)
            vertices_spa = SuccessiveProj(np.transpose(samples),K)
            _, emd_knn = compute_min_distance_pairing(np.transpose(vertex_d), vertices_knn)
            _, emd_spa = compute_min_distance_pairing(np.transpose(vertex_d), vertices_spa)
            _, emd_proj = compute_min_distance_pairing(np.transpose(vertex_d), vertices_proj)

            e_knn.append(emd_knn)
            e_proj.append(emd_proj)
            e_spa.append(emd_spa)

        error_knn.append(np.mean(e_knn))
        error_spa.append(np.mean(e_spa))
        error_proj.append(np.mean(e_proj))


    plt.plot(N_range,error_knn,"-o",label = "KNN-SPA")
    plt.plot(N_range,error_spa,"-o",label = "SPA")
    plt.plot(N_range,error_proj,"-o",label = "P-KNN-SPA")
    plt.legend()
    plt.xlabel("Sample size (n)")
    plt.ylabel("reconstruction error")
    plt.grid()
    plt.show()
if __name__ == '__main__':
    vertex = np.array([[1,2,5],[1,4,2]])

    #sigma_simulation(vertex,S_range=np.linspace(0.2,2,20))
    N_simulation(vertex,N_range=np.arange(500,2000,100))

    
    