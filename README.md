# This is the repository of our ICLR2024 paper Improved algorithm and bounds for successive projection.

Problem setup: Given some noisy observations from a K-1 diemensional simplex in d-dimension, the goal of vertex hunting is to recover the K vertices of the simplex. One common approach is the classical Successive Projection Algorithm (SPA)  (Araújo et. al 2001) which is an iterative greedy approach sensitive to outliers and affected by the dimension d. We propose a new aproach called pseudo-point SPA (ppSPA) that leverages a projection step together with a KNN denoise step to improve on the traditional SPA. We show that our method performs much better in practise and derive sharper and faster theoretical bounds.


- To reproduce the experiments from our paper run: python experiments.py


- spa.py contains the code for the orthodox spa (Araújo et. al 2001)

