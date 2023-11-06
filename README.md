# BayesFrag: Bayesian estimation of empirical seismic fragility functions

<!---
[![DOI](https://zenodo.org/badge/542139247.svg)](https://zenodo.org/badge/latestdoi/542139247)
-->

BayesFrag is an open-source Python library to perform Bayesian parameter estimation of empirical seismic fragility models. The methodology is presented in

> Bodenmann L., Baker J. , Stojadinovic B. (2023): "Accounting for ground motion uncertainty in empirical seismic fragility modeling". INCLUDE LINK

Seismic fragility functions provide a relationship between damage and a ground motion intensity measure (IM). Empirical fragility functions are estimated from damage survey data collected after past earthquakes. This is challenging, because the IM values that caused the observed damage are uncertain. BayesFrag computes the joint posterior distribution of fragility function parameters and IM values at the sites of the surveyed buildings.

![schema](https://github.com/bodlukas/BayesFrag/blob/fff6196f53253e8c1c51cde3f34c9ad39bda1e86/data/bayesfrag_schema_dark.png#gh-dark-mode-only)
![schema](https://github.com/bodlukas/BayesFrag/blob/fff6196f53253e8c1c51cde3f34c9ad39bda1e86/data/bayesfrag_schema_white.png#gh-light-mode-only)

To avoid any additional dependency on a specific library of ground motion models (GMMs), the GMM estimates are computed prior and separate to the fragility parameter estimation.

## Getting started

To allow for a smooth start, we offer three tutorials that can be opened on a hosted Jupyter notebook service (e.g., Google Colab).  

- The [first tutorial](Tutorial1.ipynb) reproduces the results of the one-dimensional example from the manuscript and explains the implementation of the proposed estimation approach: <a target="_blank" href="https://colab.research.google.com/github/bodlukas/BayesFrag/blob/main/Tutorial1.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="15"/>
</a>

- The [second tutorial](Tutorial2.ipynb) shows how the Bayesian approach can be applied to a realistic damage survey data set. The notebook can easily be modified and extended such that analysts can apply BayesFrag to their data sets of interest: <a target="_blank" href="https://colab.research.google.com/github/bodlukas/BayesFrag/blob/main/Tutorial2.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="15"/>
</a>

- The [third tutorial](Tutorial3.ipynb) explains the computation of the GMM estimates, which is done prior and separate to the fragility parameter estimation: <a target="_blank" href="https://colab.research.google.com/github/bodlukas/BayesFrag/blob/main/Tutorial3.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="15"/>
</a>

## Structure

The folder [bayesfrag](bayesfrag/) contains the main codebase of the BayesFrag package.

The folder [data](data/) contains the data used in the two tutorials.

The folder [results](results/) contains the results illustrated in the manuscript.

The notebook [figures.ipynb](figures.ipynb) reproduces the figures shown in the manuscript using utility functions from [utils_plotting.py](utils_plotting.py).

The script [main_aquila_casestudy.py](main_aquila_casestudy.py) performs all computations - from parameter estimation to post-processing - to reproduce the results presented in the paper (and stored in the folder [results](results/)). This script is computationally expensive and causes high memory demand. The computations were performed on a NVIDIA V100 tensor core GPU with 25 GB of RAM and we do not recommend to run this script on a standard personal computer.

## Installation

BayesFrag can be installed as 
```
pip install bayesfrag
```
It is good practice to do the installation in a new virtual environment, for example with [venv](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/) or with [miniconda](https://docs.conda.io/en/latest/miniconda.html). BayesFrag was tested with Python >= v3.10.

## Contact
BayesFrag is an open-source software. If you are interested in contributing to the project or have any questions, comments, or suggestions, please contact Lukas Bodenmann at bodenmann (at) ibk.baug.ethz.ch.

## Acknowledgments
We gratefully acknowledge support from the [ETH Risk Center](https://riskcenter.ethz.ch/) ("DynaRisk", Grant Nr. 395 2018-FE-213) and from the [Chair of Earthquake Engineering and Structural Dynamics](https://stojadinovic.ibk.ethz.ch/) at ETH Zurich.

## Licence
BayesFrag is released under the [BSD-3-Clause license](LICENSE).
