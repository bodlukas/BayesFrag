# Copyright (c) Lukas Bodenmann, Bozidar Stojadinovic, ETH Zurich, Switzerland
# SPDX-License-Identifier: BSD-3-Clause 

import numpy as np
from scipy import linalg

from typing import Optional
from numpy.typing import ArrayLike

from .sites import Sites
from .spatialcorrelation import SpatialCorrelationModel

class GPR(object):
    '''
    Computes the the parameters of the distribution of logIM conditional 
    on observed IMs at seismic network stations.
    '''

    def __init__(self, SCM: SpatialCorrelationModel) -> None:
        '''
        Parameters
        ----------
        SCM : SpatialCorrelationModel
            Employed Spatial correlation model
        '''
        self.SCM = SCM

    def getCov(self, sites1: Sites, sites2: Optional[Sites] = None, full_cov=True) -> ArrayLike:
        '''
        Computes covariance matrix of the multivariate normal distribution 
        of logIM values conditional on rupture characteristics.

        Parameters
        ----------
        sites1 : Sites
            Primary sites for which to compute covariance matrix.
        sites2 : Sites, optional
            Secondary sites. If provided, computes covariance matrix 
            between primary and secondary sites. 
        full_cov : bool, defaults to True
            if True computes full covariance matrix
            if False computes only the diagonal

        Returns
        -------
        cov : ArrayLike
            Covariance matrix
        '''
        if full_cov:
            corr_mat = self.SCM.get_correlation_matrix(sites1, sites2)
            if sites2 is None: 
                # Between-event term
                cov = np.matmul(np.atleast_2d(sites1.tau_logIM).T,
                    np.atleast_2d(sites1.tau_logIM))
                # Within-event term
                cov += (np.matmul(np.atleast_2d(sites1.phi_logIM).T,
                    np.atleast_2d(sites1.phi_logIM)) * corr_mat)
            else:
                # Between-event term
                cov = np.matmul(np.atleast_2d(sites1.tau_logIM).T,
                    np.atleast_2d(sites2.tau_logIM))
                # Within-event term
                cov += (np.matmul(np.atleast_2d(sites1.phi_logIM).T,
                    np.atleast_2d(sites2.phi_logIM)) * corr_mat)
        else: # return diagonal of covariance matrix: tau**2 + phi**2
            cov = np.square(np.atleast_1d(sites1.tau_logIM))
            cov += np.square(np.atleast_1d(sites1.phi_logIM))
        return cov

    def fit(self, sites: Sites, obs_logIM: ArrayLike, jitter: float = 1e-6) -> None:
        '''
        Compute and cache variables required for predictions

        Parameters
        ----------
        sites : Sites
            Site collection for seismic network stations.
        obs_logIM : ArrayLike
            Observed logIM at seismic stations
        jitter : float, defaults to 1e-6
            Add for numerical stability
        '''

        Sigma_SS = self.getCov(sites)
        Sigma_SS = Sigma_SS + np.eye(sites.n_sites)*jitter
        self._L = np.linalg.cholesky(Sigma_SS)

        residuals = obs_logIM - np.atleast_1d(sites.mu_logIM)
        self.sites = sites
        self._alpha = linalg.cho_solve((self._L, True), residuals)

    def predict(self, sites_new: Sites, full_cov: bool=True) -> tuple[ArrayLike, ArrayLike]:
        '''
        Compute the parameters of the normal distribution of logIM 
        at target sites conditioned on observations at station sites

        Parameters
        ----------
        sites_new : Sites
            Site collection for target sites
        full_cov : bool, defaults to True
            if True computes full covariance matrix
            if False computes only the diagonal

        Returns
        ----------
        mu_T_S : ArrayLike, dimension (n_sites,)
            Mean logIM values conditional on station data. 
            Mean vector of conditional normal distribution.
        Sigma_TT_S : ArrayLike
            if full_cov is True: Covariance matrix of conditional
            normal distribution, dimension (n_sites, n_sites)
            if full_cov is False: Diagonal of covariance matrix,
            dimension (n_sites,)
        '''
        Sigma_ST = self.getCov(self.sites, sites_new, full_cov=True)
        Sigma_TT = self.getCov(sites_new, full_cov=full_cov)
        A = linalg.solve_triangular(self._L, Sigma_ST, lower=True)
        mu_T_S = (np.atleast_1d(sites_new.mu_logIM) +
                (np.matmul(Sigma_ST.T, self._alpha)) )
        if full_cov:
            Sigma_TT_S = Sigma_TT - np.matmul(A.T, A)
        else:
            Sigma_TT_S = Sigma_TT - np.einsum("ij,ji->i", A.T, A)
        return mu_T_S, Sigma_TT_S