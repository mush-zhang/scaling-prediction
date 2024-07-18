# scaling-prediction

This repository contains code for our paper [**From Feature Selection to Resource Prediction: A Survey of Commonly Applied Workflows and Techniques**](https://anonymous.4open.science/r/scaling-prediction-C526/ScalingPerformanceComputation_extended.pdf)

We examine the state-of-the-art strategies for  the three-step pipeline for workload scaling prediction: feature selection, workload similarity, and performance prediction, with the goal to identify which techniques work best in practice. Our experimental results reveal that while no universal solution exists for the prediction pipeline, certain best practices can improve prediction performance and reduce computation overhead. Based on our results, we outline important topics for future work that will benefit ML driven recommendation systems for resource allocation.

## Run Notebooks

On Ubuntu 22.04.

```bash
    $ wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh 

    $ bash Miniforge3-Linux-x86_64.sh 

    $ eval "$([CONDA_PREFIX]/miniforge3/bin/conda shell.bash hook)" 

    $ conda install sdt-python seaborn tslearn ruptures jupyterlab
```

Run experiments with jupter lab. The names of the notebooks include the numberings of the tables/figures they contribute to.
