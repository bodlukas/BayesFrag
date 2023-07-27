import pandas as pd
import numpy as onp
from scipy import linalg

def prepare_sites(df, im_args):
  sites = {'X': df[['x','y']].values,
            'vs30': df['vs30'].values}
  gmm_res = df[[im_args['mu'], im_args['tau'], im_args['phi']]].to_dict(orient='list')
  return sites, gmm_res

def get_euclidean_distance_matrix(sites1, sites2 = None, full_cov = True):
  X1 = sites1['X']
  if full_cov:
    if sites2 is None: X2 = X1
    else: X2 = sites2['X']
    sq_dist = onp.sum(onp.stack([
        onp.square(onp.reshape(X1[:,i], [-1,1]) - onp.reshape(X2[:,i], [1,-1]))
                for i in range(X1.shape[1])]),axis=0)
    sq_dist = onp.clip(sq_dist, 0, onp.inf)
    dist = onp.sqrt(sq_dist)
  else:
    dist = onp.zeros(X1.shape[0])
  return dist

# Correlation model of Esposito and Iervolino (2012)
def get_correlation_matrix_EI2012(sites1, sites2=None, full_cov=True):
    corr_range = 11.3
    dist_mat_E = get_euclidean_distance_matrix(sites1, sites2, full_cov)
    return onp.exp( - 3 * dist_mat_E / corr_range )

# Compute Covariance Matrix
def getCov(args_im, sites1, gmm_res1, sites2 = None, gmm_res2 = None, full_cov=True):
  '''
  Computes covariance matrix 
  sites1    : coordinates of the primary sites
  gmm_res1  : pre-computed results of single-site GMM for the primary sites 

  if sites2 or gmm_res2 are None, it computes the covariance matrix between each
  site in sites1 to every other site in sites1. 

  if sites2 and gmm_res2 are provided, it computes the covariance matrix between
  each site in sites1 to every site in sites2.

  if full_cov is False, it computes only the diagonal of the covariance matrix.
  '''
  if full_cov:
    corr_mat = get_correlation_matrix_EI2012(sites1, sites2)
    if gmm_res2 is None: 
      # Between-event term
      cov = onp.matmul(onp.atleast_2d(gmm_res1[args_im['tau']]).T,
            onp.atleast_2d(gmm_res1[args_im['tau']]))
      # Within-event term
      cov += (onp.matmul(onp.atleast_2d(gmm_res1[args_im['phi']]).T,
            onp.atleast_2d(gmm_res1[args_im['phi']])) * corr_mat)
    else:
      # Between-event term
      cov = onp.matmul(onp.atleast_2d(gmm_res1[args_im['tau']]).T,
            onp.atleast_2d(gmm_res2[args_im['tau']]))
      # Within-event term
      cov += (onp.matmul(onp.atleast_2d(gmm_res1[args_im['phi']]).T,
            onp.atleast_2d(gmm_res2[args_im['phi']])) * corr_mat)
  else: # return diagonal of covariance matrix: tau**2 + phi**2
    cov = onp.square(onp.atleast_1d(gmm_res1[args_im['tau']]))
    cov += onp.square(onp.atleast_1d(gmm_res1[args_im['phi']]))
  return cov

# Compute mean vector (mean of logIM)
def getMean(args_im, gmm_res):
  return onp.atleast_1d(gmm_res[args_im['mu']])


class GPR(object):
  '''
  This object is used to compute the distribution of logIM conditional on 
  observed IMs at seismic network stations.

  The mean function and the kernel are objects similar to getMean and getCov 
  from above.
  '''

  def __init__(self, meanfunction, kernel, args_im):
    self.kernel = kernel
    self.meanfunction = meanfunction
    self.args_im = args_im

  def fit(self, sites, gmm_res, y, jitter=1e-6):
    '''
    Compute and cache variables required for predictions

    sites : site information for seismic stations
    gmm_res : pre-computed GMM results for seismic stations and rupture

    '''
    N = sites['X'].shape[0]
    K = self.kernel(self.args_im, sites, gmm_res)
    K = K + onp.eye(N)*jitter
    self._L = onp.linalg.cholesky(K)

    m = self.meanfunction(self.args_im, gmm_res)
    residuals = y - m
    self.sites = sites
    self.gmm_res = gmm_res
    self._alpha = linalg.cho_solve((self._L,True), residuals)

  def predict(self, sites_new, gmm_res_new, full_cov=True):
    '''
    Compute the distribution of logIM condition on recordings at 
    target sites

    sites_new : site information for target sites
    gmm_res_new : pre-computed GMM results for target sites and rupture
    full_cov : if True computes entire covariance matrix, otherwise only return
                the diagonal 
    
    '''
    Kmn = self.kernel(self.args_im, self.sites, self.gmm_res,
                      sites_new, gmm_res_new, full_cov=True)
    Knn = self.kernel(self.args_im, sites_new, gmm_res_new, full_cov=full_cov)
    A = linalg.solve_triangular(self._L, Kmn, lower=True)
    fmean = (self.meanfunction(self.args_im, gmm_res_new) +
             (onp.matmul(Kmn.T, self._alpha)) )
    if full_cov:
      fvar = Knn - onp.matmul(A.T, A)
    else:
      fvar = Knn - onp.einsum("ij,ji->i", A.T, A)
      # fvar = Knn - onp.sum(A**2, axis=0)
    return fmean, fvar