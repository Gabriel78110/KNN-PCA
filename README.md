# Improved Algorithm and Bounds for Successive Projection

This is the repository for our paper <span style="color:blue; font-weight:bold;">"Improved algorithm and bounds for successive projection"</span> published at **<span style="color:blue;">ICLR 2024</span>**.

## Problem Setup

Given noisy observations from a (K-1)-dimensional simplex in d-dimensions, the goal of vertex hunting is to recover the K vertices of the simplex. We propose a new approach called Pseudo-point SPA (ppSPA), which leverages a projection step together with a KNN denoise step to improve on the traditional SPA. Our method performs better in practice and yields sharper and faster theoretical bounds.


![Example Image](Results-experiments/triangle.png)


## Reproducing Experiments

To reproduce the experiments from our paper, run:

```bash
python experiments.py
```

For more details, refer to our [ICLR 2024 paper](link-to-paper).

