import numpy as np
from scipy import linalg

from modules.utils import Sites
from modules.spatialcorrelation import SpatialCorrelationModel

class GPR(object):
    '''
    This object is used to compute the distribution of logIM conditional on 
    observed IMs at seismic network stations.

    The mean and the covariance functions are functions similar to getMean and 
    getCov from above.
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
                if False computs only the diagonal
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

    def fit(self, sites, obs_logIM, jitter=1e-6):
        '''
        Compute and cache variables required for predictions

        sites : site information for seismic stations
        obs_logIM : observed logIM at seismic stations

        '''

        K = self.getCov(sites)
        K = K + np.eye(sites.n_sites)*jitter
        self._L = np.linalg.cholesky(K)

        m = np.atleast_1d(sites.mu_logIM)
        residuals = obs_logIM - m
        self.sites = sites
        self._alpha = linalg.cho_solve((self._L,True), residuals)

    def predict(self, sites_new, full_cov=True):
        '''
        Compute the distribution of logIM at target sites conditioned on 
        observations at station sites

        sites_new : site information for target sites
        full_cov : if True computes entire covariance matrix, otherwise only return
                    the diagonal 
        
        '''
        Kmn = self.getCov(self.sites, sites_new, full_cov=True)
        Knn = self.getCov(sites_new, full_cov=full_cov)
        A = linalg.solve_triangular(self._L, Kmn, lower=True)
        fmean = (np.atleast_1d(sites_new.mu_logIM) +
                (np.matmul(Kmn.T, self._alpha)) )
        if full_cov:
            fvar = Knn - np.matmul(A.T, A)
        else:
            fvar = Knn - np.einsum("ij,ji->i", A.T, A)
        return fmean, fvar