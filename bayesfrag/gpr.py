import numpy as np
from scipy import linalg

from .sites import Sites
from .spatialcorrelation import SpatialCorrelationModel

class GPR(object):
    '''
    Computes the distribution of logIM conditional on 
    observed IMs at seismic network stations.
    '''

    def __init__(self, SCM: SpatialCorrelationModel):
        '''
        Args:
            SCM (SpatialCorrelationModel): Correlation model
        '''
        self.SCM = SCM

    def getCov(self, sites1: Sites, sites2 = None, full_cov=True):
        '''
        Computes covariance matrix
        if sites2 is None, it computes the covariance matrix 
        between each site in sites1 to every other site in sites1. 
        if sites2 is provided, it computes the covariance matrix 
        between each site in sites1 to each site in sites2.

        Args:

            sites1 (Sites): Primary sites

            sites2 (Sites, Optional): Secondary sites

            full_cov (bool): 
                if True computes full covariance matrix
                if False computes only the diagonal
                defaults to True
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

    def fit(self, sites: Sites, obs_logIM, jitter=1e-6):
        '''
        Compute and cache variables required for predictions

        Args:
            sites (Sites): Site collection for seismic stations
            
            obs_logIM (ArrayLike): Observed logIM at seismic stations

            jitter (float, Optional): Add for numerical stability

        '''

        Sigma_SS = self.getCov(sites)
        Sigma_SS = Sigma_SS + np.eye(sites.n_sites)*jitter
        self._L = np.linalg.cholesky(Sigma_SS)

        residuals = obs_logIM - np.atleast_1d(sites.mu_logIM)
        self.sites = sites
        self._alpha = linalg.cho_solve((self._L, True), residuals)

    def predict(self, sites_new, full_cov=True):
        '''
        Compute the distribution of logIM at target sites conditioned on 
        observations at station sites

        Args:
            sites_new : Site collection for target sites
            
            full_cov : if True computes entire covariance matrix, otherwise only 
                        the diagonal 
        
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