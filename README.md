# BayesFrag: Bayesian estimation of empirical seismic fragility functions

This repository is currently under development. The links do not yet work.

<!---
[![DOI](https://zenodo.org/badge/542139247.svg)](https://zenodo.org/badge/latestdoi/542139247)
-->

This repository contains the code to perform Bayesian parameter estimation for empirical seismic fragility models. The methodology is presented in :

> Bodenmann L., Baker J. , Stojadinovic B. (2023): "Accounting for ground motion uncertainty in empirical seismic fragility modeling". INCLUDE LINK

<!---
![alt text](https://github.com/bodlukas/ground-motion-correlation-bayes/blob/main/data/corr_schema_dark.png#gh-dark-mode-only)
![alt text](https://github.com/bodlukas/ground-motion-correlation-bayes/blob/main/data/corr_schema_bright.png#gh-light-mode-only)
-->

## Getting started

The notebook ... reproduces the results of the one-dimensional example from the manuscript and explains the implementation of the proposed estimation approach. 

The notebook ... shows how the Bayesian approach can be applied to a realistic damage survey data set. 

Both notebooks can be opened on a hosted Jupyter notebook service (e.g., Google Colab). This allows for a smooth start, because it does not require any local python setup. 

To avoid any additional dependency on a specific ground motion model (GMM) library, the GMM computions are performed prior and separate to the fragility parameter estimation. For the example used in the second tutorial, the notebook ... explains these computations.  

## Structure

The folder [data](data/) contains the data used in the two tutorials.

The folder [results](results/) contains the results illustrated in the manuscript.

The folder [modules](modules/) contains the main scripts used and explained in the tutorials.

The notebook [figures.ipynb](figures.ipynb) reproduces the figures shown in the manuscript using utility functions from [utils_plotting.py](utils_plotting.py).

The required packages are specified in [environment.yml](environment.yml).

## Installation

We performed model estimation on Linux and did not test other opearting systems. To perform model estimation on your local machine, you can set up a Windows Subsystem for Linux ([WSL](https://learn.microsoft.com/en-us/windows/wsl/install)). Then install [miniconda](https://docs.conda.io/en/latest/miniconda.html) and create an environment as `conda env create -f environment.yml`.
