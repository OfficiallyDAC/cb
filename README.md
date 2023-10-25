# CB

This repository contains the source code for the paper _Extracting the Multiscale Causal Backbone of Brain Dynamics_.

# Dependencies
In order to properly run the provided code, please install the dependencies reported in the file `cb.yml`.

As per [conda documentation](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file), you can create the conda environment _CB_ by running the following command in your terminal:
```shell
    conda env create -f cb.yml
```

Once the environment has been successfully created, if using VS Code, then type
```shell
    code .
```
and select the CB environment from those available.
Otherwise, please activate it
```shell
    conda activate CB
```
and run the provided IPython notebooks.
To deactivate the environment, simply run 
```shell
    conda deactivate
```

# Data
Since the size of the files is large, it is not feasible to upload the data on this repository.
However, data are available upon requests.
The `data` dir is supposed to be in the same folder of `notebooks`.

# Source code
The `code` folder contains the implementation of the methods used in the paper, along with utils and evaluation metrics.

# IPython notebooks
- The `0.*` and `1.*` notebooks reproduce our case study on fMRI data.
- The `2.*` notebooks reproduce the empirical assessment on synthetic data.
