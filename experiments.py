from vertexH import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from pyemd import emd
import seaborn as sns
from projection import get_projected_points

import numpy as np
import pandas as pd

M_ = 5

"---------------------------- SIMULATION DIMENSION ------------------------------------"

def d_simulation(vertex,D=10,S=1,N_sample=1000):
    error_knn, error_spa, error_proj_knn, error_proj = [], [], [], []
    K = vertex.shape[1]

    for d in np.arange(2,D):
        e_knn, e_spa, e_proj_knn, e_proj = [], [], [], []
        for _ in range(5):
            A = np.random.randint(2,size=(d,2))
            vertex_d = np.dot(A,vertex)
            samples = np.dot(A,np.transpose(get_sample(N_sample = N_sample, vertex=vertex)))
            samples +=  np.random.normal(scale=S, size=(d,N_sample))

            samples_ = get_projected_points(samples)
            eps_proj = find_max_distance_to_average(np.transpose(samples_))/M_
            eps_knn = find_max_distance_to_average(np.transpose(samples))/M_
            X_proj_knn = compute_points_within_epsilon(np.transpose(samples_),epsilon=eps_proj,N=4,t=3,d=d)
            X_knn = compute_points_within_epsilon(np.transpose(samples),epsilon=eps_knn,N=4,t=3,d=d)
            
            vertices_proj_knn = SuccessiveProj(X_proj_knn,K)
            vertices_proj = SuccessiveProj(np.transpose(samples_),K)
            vertices_knn = SuccessiveProj(X_knn,K)
            vertices_spa = SuccessiveProj(np.transpose(samples),K)

            _, emd_knn = compute_min_distance_pairing(np.transpose(vertex_d), vertices_knn)
            _, emd_spa = compute_min_distance_pairing(np.transpose(vertex_d), vertices_spa)
            _, emd_proj_knn = compute_min_distance_pairing(np.transpose(vertex_d), vertices_proj_knn)
            _, emd_proj = compute_min_distance_pairing(np.transpose(vertex_d), vertices_proj)

            e_knn.append(emd_knn)
            e_proj_knn.append(emd_proj_knn)
            e_spa.append(emd_spa)
            e_proj.append(emd_proj)

        error_knn.append(np.mean(e_knn))
        error_spa.append(np.mean(e_spa))
        error_proj_knn.append(np.mean(e_proj_knn))
        error_proj.append(np.mean(e_proj))


    return error_knn, error_spa, error_proj, error_proj_knn

    # plt.figure(figsize=(8,5))
    # plt.plot(np.arange(2,D),error_knn,"-o",label = "KNN-SPA")
    # plt.plot(np.arange(2,D),error_proj,"-o",label = "P-SPA")
    # plt.plot(np.arange(2,D),error_spa,"-o",label = "SPA")
    # plt.plot(np.arange(2,D),error_proj_knn,"-o",label = "P-KNN-SPA")
    # plt.legend()
    # plt.xlabel("dimension (d)")
    # plt.ylabel("reconstruction error")
    # plt.grid()
    # plt.savefig("d.png")
    # plt.show()

"---------------------------- SIMULATION NOISE ------------------------------------"


def sigma_simulation(vertex,S_range,d=4,N_sample=1000):
    error_knn, error_spa, error_proj_knn, error_proj = [], [], [], []
    K = vertex.shape[1]

    for S in S_range:
        e_knn, e_spa, e_proj_knn, e_proj = [], [], [], []
        for _ in range(5):
            #A = np.random.randint(2,size=(d,2))
            A = np.eye(d,2)
            vertex_d = np.dot(A,vertex)
            samples = np.dot(A,np.transpose(get_sample(N_sample = N_sample, vertex=vertex)))
            samples +=  np.random.normal(scale=S, size=(d,N_sample))

            samples_ = get_projected_points(samples)
            eps_proj = find_max_distance_to_average(np.transpose(samples_))/M_
            eps_knn = find_max_distance_to_average(np.transpose(samples))/M_
            X_proj_knn = compute_points_within_epsilon(np.transpose(samples_),epsilon=eps_proj,N=4,t=3,d=d)
            X_knn = compute_points_within_epsilon(np.transpose(samples),epsilon=eps_knn,N=4,t=3,d=d)
            
            vertices_proj_knn = SuccessiveProj(X_proj_knn,K)
            vertices_proj = SuccessiveProj(np.transpose(samples_),K)
            vertices_knn = SuccessiveProj(X_knn,K)
            vertices_spa = SuccessiveProj(np.transpose(samples),K)

            _, emd_knn = compute_min_distance_pairing(np.transpose(vertex_d), vertices_knn)
            _, emd_spa = compute_min_distance_pairing(np.transpose(vertex_d), vertices_spa)
            _, emd_proj_knn = compute_min_distance_pairing(np.transpose(vertex_d), vertices_proj_knn)
            _, emd_proj = compute_min_distance_pairing(np.transpose(vertex_d), vertices_proj)

            e_knn.append(emd_knn)
            e_proj_knn.append(emd_proj_knn)
            e_spa.append(emd_spa)
            e_proj.append(emd_proj)

        error_knn.append(np.mean(e_knn))
        error_spa.append(np.mean(e_spa))
        error_proj.append(np.mean(e_proj))
        error_proj_knn.append(np.mean(e_proj_knn))

    return error_knn, error_spa, error_proj, error_proj_knn
    # plt.figure(figsize=(8,5))
    # plt.plot(S_range,error_knn,"-o",label = "KNN-SPA")
    # plt.plot(S_range,error_proj,"-o",label = "P-SPA")
    # plt.plot(S_range,error_spa,"-o",label = "SPA")
    # plt.plot(S_range,error_proj_knn,"-o",label = "P-KNN-SPA")
    # plt.legend()
    # plt.xlabel("Noise level (sigma)")
    # plt.ylabel("reconstruction error")
    # plt.grid()
    # plt.savefig("S.png")
    # plt.show()

"---------------------------- SIMULATION SAMPLE SIZE ------------------------------------"

def N_simulation(vertex,N_range,d=4,S=1):
    error_knn, error_spa, error_proj, error_proj_knn = [], [], [], []
    K = vertex.shape[1]

    for N_ in N_range:
        e_knn, e_spa, e_proj, e_proj_knn = [], [], [], []
        for _ in range(5):
            #A = np.random.randint(2,size=(d,2))
            A = np.eye(d,2)
            vertex_d = np.dot(A,vertex)
            samples = np.dot(A,np.transpose(get_sample(N_sample = N_, vertex=vertex)))
            samples +=  np.random.normal(scale=S, size=(d,N_))

            samples_ = get_projected_points(samples)
            eps_proj = find_max_distance_to_average(np.transpose(samples_))/M_
            eps_knn = find_max_distance_to_average(np.transpose(samples))/M_
            X_proj_knn = compute_points_within_epsilon(np.transpose(samples_),epsilon=eps_proj,N=4,t=3,d=d)
            X_knn = compute_points_within_epsilon(np.transpose(samples),epsilon=eps_knn,N=4,t=3,d=d)
            
            vertices_proj_knn = SuccessiveProj(X_proj_knn,K)
            vertices_proj = SuccessiveProj(np.transpose(samples_),K)
            vertices_knn = SuccessiveProj(X_knn,K)
            vertices_spa = SuccessiveProj(np.transpose(samples),K)

            _, emd_knn = compute_min_distance_pairing(np.transpose(vertex_d), vertices_knn)
            _, emd_spa = compute_min_distance_pairing(np.transpose(vertex_d), vertices_spa)
            _, emd_proj = compute_min_distance_pairing(np.transpose(vertex_d), vertices_proj)
            _, emd_proj_knn = compute_min_distance_pairing(np.transpose(vertex_d), vertices_proj_knn)

            e_knn.append(emd_knn)
            e_proj.append(emd_proj)
            e_proj_knn.append(emd_proj_knn)
            e_spa.append(emd_spa)

        error_knn.append(np.mean(e_knn))
        error_spa.append(np.mean(e_spa))
        error_proj.append(np.mean(e_proj))
        error_proj_knn.append(np.mean(e_proj_knn))

    return error_knn, error_spa, error_proj, error_proj_knn
    # plt.figure(figsize=(8,5))
    # plt.plot(N_range,error_knn,"-o",label = "KNN-SPA")
    # plt.plot(N_range,error_proj,"-o",label = "P-SPA")
    # plt.plot(N_range,error_spa,"-o",label = "SPA")
    # plt.plot(N_range,error_proj_knn,"-o",label = "P-KNN-SPA")
    # plt.legend()
    # plt.xlabel("Sample size (n)")
    # plt.ylabel("reconstruction error")
    # plt.grid()
    # plt.savefig("N.png")
    # plt.show()



"---------------------------- SIMULATION ESTIMATING VARIANCE ------------------------------------"

def estimator_var(vertex,S_range,N_sample=1000,d=10):
    K = vertex.shape[1]
    var_diff = []
    A = np.random.randint(2,size=(d,2))
    for S in S_range:
        samples = np.dot(A,np.transpose(get_sample(N_sample = N_sample, vertex=vertex)))
        samples +=  np.random.normal(scale=S, size=(d,N_sample))
        samples_ , X_bar, mean = get_projected_points(samples,return_X=True)
        res = samples - samples_
        var_diff.append((np.sum(np.linalg.norm(res,axis=0)**2)/((N_sample-1)*(d-K+1))) - S**2)

    plt.plot(S_range,var_diff,"-o")
    plt.xlabel("True standard deviation (sigma)")
    plt.ylabel("|sigma - sigma_hat|")
    plt.grid()
    plt.show()


"---------------------------- SIMULATION MEAN CHI2 ------------------------------------"

def max_chi2(d_range,N=1000,N_iter=500):
    
    M = []
    df = pd.DataFrame({"df":[], "M_n":[]})
    for i, d in enumerate(d_range):

        sample = np.random.chisquare(df=d,size=(N_iter,N))
        M_n = np.mean(np.max(sample,axis=1))
        M.append(round(M_n,2))
        df1 = pd.DataFrame({"df":d, "M_n":round(M_n,2)},index=[i])
        df = pd.concat([df,df1])

    df.T.to_csv("Results-experiments/table_chi2.csv",header=False)
    plt.plot(d_range,M,"-o")
    plt.xlabel("Df")
    plt.ylabel("M_n")
    plt.axhline(y = 2*np.log(N), color = 'r', linestyle = '-',label="2.log(1000)")
    plt.legend()
    plt.grid()
    plt.show()

"-----------------------------------------------------------------------------------"

if __name__ == '__main__':
    N_range = np.arange(500,2000,100)
    D_range = np.arange(2,10)
    S_range = np.linspace(0.2,2,20)
    vertex = np.array([[1,2,5],[1,4,2]])
    d_range = [2, 4, 6, 8, 10, 12]
    #max_chi2(d_range = d_range)
    error_knn_d, error_spa_d, error_proj_d, error_proj_knn_d = d_simulation(vertex)
    error_knn_s, error_spa_s, error_proj_s, error_proj_knn_s = sigma_simulation(vertex,S_range)
    error_knn_n, error_spa_n, error_proj_n, error_proj_knn_n = N_simulation(vertex,N_range)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10,3))
    ax1.plot(D_range,error_knn_d,"-o",label = "PPSPA (step 2)")
    ax1.plot(D_range,error_spa_d,"-o",label = "SPA")
    ax1.plot(D_range,error_proj_d,"-o",label = "PPSPA (step 1)")
    ax1.plot(D_range,error_proj_knn_d,"-o",label = "PPSPA")
    ax1.set_xlabel("Dimension (d)")

    ax2.plot(S_range,error_knn_s,"-o",label = "PPSPA (step 2)")
    ax2.plot(S_range,error_spa_s,"-o",label = "SPA")
    ax2.plot(S_range,error_proj_s,"-o",label = "PPSPA (step 1)")
    ax2.plot(S_range,error_proj_knn_s,"-o",label = "PPSPA")
    ax2.set_xlabel("Noise level (sigma)")

    ax3.plot(N_range,error_knn_n,"-o",label = "PPSPA (step 2)")
    ax3.plot(N_range,error_spa_n,"-o",label = "SPA")
    ax3.plot(N_range,error_proj_n,"-o",label = "PPSPA (step 1)")
    ax3.plot(N_range,error_proj_knn_n,"-o",label = "PPSPA")
    ax3.legend()
    ax3.set_xlabel("Sample size (n)")

    fig.text(0.0001, 0.5, 'Reconstruction error', va='center', rotation='vertical')
    plt.tight_layout()
    plt.savefig("Experiments_subplots.png")
    plt.show()



    # plt.plot(N_range,error_proj,"-o",label = "P-SPA")
    # plt.plot(N_range,error_spa,"-o",label = "SPA")
    # plt.plot(N_range,error_proj_knn,"-o",label = "P-KNN-SPA")
    # plt.legend()
    # plt.xlabel("Sample size (n)")
    # plt.ylabel("reconstruction error")
    # plt.grid()
    
    # sigma_simulation(vertex,S_range=np.linspace(0.2,2,20))
    # N_simulation(vertex,N_range=np.arange(500,2000,100))
    #estimator_var(vertex,np.linspace(0.2,2,20))

    
    