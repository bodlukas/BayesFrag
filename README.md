# BayesFrag: Bayesian estimation of empirical seismic fragility functions

This repository is currently under development. The links do not yet work.

<!---
[![DOI](https://zenodo.org/badge/542139247.svg)](https://zenodo.org/badge/latestdoi/542139247)
-->

This repository contains the code to perform Bayesian parameter estimation for empirical seismic fragility models. The methodology is presented in :

> Bodenmann L., Baker J. , Stojadinovic B. (2023): "Accounting for ground motion uncertainty in empirical seismic fragility modeling". INCLUDE LINK

![alt text](https://github.com/bodlukas/empirical-fragility-bayes/blob/main/data/bayesfrag-schema_darks.png#gh-dark-mode-only)
![alt text](https://github.com/bodlukas/empirical-fragility-bayes/blob/main/data/bayesfrag-schema_whites.png#gh-light-mode-only)

## Getting started

To allow for a smooth start, we offer three tutorials that can be opened on a hosted Jupyter notebook service (e.g., Google Colab).  

- The [first tutorial](Tutorial1.ipynb) reproduces the results of the one-dimensional example from the manuscript and explains the implementation of the proposed estimation approach. 

- The [second tutorial](Tutorial2.ipynb) shows how the Bayesian approach can be applied to a realistic damage survey data set. 

- To avoid any additional dependency on a specific ground motion model (GMM) library, the GMM computions are performed prior and separate to the fragility parameter estimation. The [third tutorial](Tutorial3.ipynb) explains these computations.  

## Structure

The folder [bayesfrag](bayesfrag/) contains the main codebase that is used and explained in the tutorials.

The folder [data](data/) contains the data used in the two tutorials.

The folder [results](results/) contains the results illustrated in the manuscript.

The notebook [figures.ipynb](figures.ipynb) reproduces the figures shown in the manuscript using utility functions from [utils_plotting.py](utils_plotting.py).

The script [main_aquila_casestudy.py](main_aquila_casestudy.py) performs all computations - from parameter estimation to post-processing - to reproduce the results presented in the paper (and stored in the folder [results/aquila](results/aquila/)). This script is computationally expensive and causes high memory demand. The computations were performed on a NVIDIA V100 tensor core GPU with 25 GB of RAM and we do not recommend to run this script on a standard personal computer.

The required packages are specified in [environment.yml](environment.yml).

## Installation

BayesFrag can be installed by using 
```python
pip install bayesfrag
```
We performed model estimation on Linux and did not test other opearting systems. To perform model estimation on your local machine, you can set up a Windows Subsystem for Linux ([WSL](https://learn.microsoft.com/en-us/windows/wsl/install)). Then install [miniconda](https://docs.conda.io/en/latest/miniconda.html) and create an environment as `conda env create -f environment.yml`.
