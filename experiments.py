from vertexH import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from pyemd import emd
import seaborn as sns
from projection import get_projected_points

import numpy as np


def eps_exp(vertex, samples, eps_range):
    error1 = []
    #error2 = []
    for eps in eps_range:
        X_tilde = compute_points_within_epsilon(samples,epsilon=eps,N=4,t=3)
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


def t_exp(vertex,samples,t_range):
    error1 = []
    #error2 = []
    # eps = knn(samples)
    eps = 0.1
    for t in t_range:
        X_tilde = compute_points_within_epsilon(samples,eps,N=3,t=t)
        vertices_knn = SuccessiveProj(X_tilde,K=3)
        vertices_spa = SuccessiveProj(samples,K=3)
        _, emd1 = compute_min_distance_pairing(np.transpose(vertex), vertices_knn)
        #_, emd2 = compute_min_distance_pairing(np.transpose(vertex), vertices_spa)
        error1.append(emd1)
        #error2.append(emd2)
    plt.plot(t_range,error1, label = "KNN-SPA")
    #plt.plot(N_range,error2, label = "SPA")
    plt.xlabel("N")
    plt.ylabel("sum of euclidean distances")
    plt.legend()
    plt.show()




if __name__ == '__main__':

    vertex = np.array([[1,2,5],[1,4,2]])
    N_sample = 1000
    K = vertex.shape[1]
    S = 1
    D = 10
    error_knn, error_spa, error_proj = [], [], []


    for d in np.arange(3,D):
        e_knn, e_spa, e_proj = [], [], []
        for _ in range(5):
            A = np.random.randint(2,size=(d,2))
            vertex_d = np.dot(A,vertex)
            #vertex_d = np.dot(np.eye(d,2),vertex)
            samples = np.dot(A,np.transpose(get_sample(N_sample = N_sample, vertex=vertex)))
            #samples = np.dot(np.eye(d,2),np.transpose(get_sample(N_sample = N_sample, vertex=vertex,sigma=0.2)))
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


    plt.plot(np.arange(3,D),error_knn,"-o",label = "KNN-SPA")
    plt.plot(np.arange(3,D),error_spa,"-o",label = "SPA")
    plt.plot(np.arange(3,D),error_proj,"-o",label = "P-KNN-SPA")
    plt.legend()
    plt.xlabel("dimension (d)")
    plt.ylabel("reconstruction error")
    plt.grid()
    plt.show()
        #vertices_knn = np.concatenate((vertices_knn, vertices_knn[0].reshape(1,-1)),axis=0)
        #print(compute_min_distance_pairing(np.transpose(vertex),vertices_knn))
        # vertices_spa = np.concatenate((vertices_spa, vertices_spa[0].reshape(1,-1)),axis=0)
        # plot_simplex(vertex, samples)
        # plt.plot(vertices_knn[:,0], vertices_knn[:,1],"b--", marker='o',label="KNN-SPA reconstruction")
        # plt.plot(vertices_spa[:,0], vertices_spa[:,1],"g--", marker='o',label="SPA reconstruction")
        # plt.legend()
        # plt.title(f"eps =")
        # plt.show()
















    


# if __name__ == '__main__':
#     vertex = np.array([[1,2,5],[1,4,2]])
#     t_range = np.arange(3,10)
#     eps_range = np.linspace(0.1,2,50)
#     samples = np.load("samples_smallvar.npy")

#     #eps_exp(vertex,samples,eps_range=eps_range)
#     t_exp(vertex,samples,t_range=t_range)
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
    