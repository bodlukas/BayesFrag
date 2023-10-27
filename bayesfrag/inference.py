# ADD COPYRIGHT AND LICENSE

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO

from bayesfrag.utils import get_pmf_ds_logIM
from bayesfrag.postprocess import Posterior, PointEstimates

# def Model_MCMC(mu, L, ds, bc, parampriors):
#     '''
#     Prior generative model encoded in Numpyro

#     args:
#         mu (ArrayLike): Mean of logIM at survey sites conditional on station data
#             dimension: (n_sites, )

#         L (ArrayLike): Lower Cholesky decomposition of the covariance matrix of logIM
#                         at survey sites conditional on station data
#             dimension: (n_sites, n_sites)

#         ds (ArrayLike): Observed damage states (encoded with integers)
#             dimension: (n_sites,)

#         bc (ArrayLike): Observed building classes (encoded with integers)
#             dimension: (n_sites,)

#         parampriors (dict): Prior distributions for fragility function parameters. See
#                             example of default_priors above.
#     '''

#     N = ds.shape[0] # Number of data points

#     # Prior sampling distributions of fragility function parameters
#     eta1 = numpyro.sample('eta1', parampriors['eta1']) 
#     deltas = numpyro.sample('deltas', parampriors['deltas']) 
#     beta = numpyro.sample('beta', parampriors['beta']) 

#     # Prior sampling distribution of z
#     z = numpyro.sample('z', dist.Normal(jnp.zeros(N), 1) )

#     # Transform z to logIM at survey sites
#     logIM = mu + jnp.matmul(L, z)

#     # Compute damage state probabilities
#     p = get_pmf_ds_logIM(logIM, bc, eta1, deltas, beta).T

#     # Likelihood
#     numpyro.sample('obs', dist.Categorical(probs = p), obs = ds)

#---------
# Bayesian inference with uncertain IM values
#---------

class Bayesian_MCMC(object):
  '''
  Wrapper around Numpyro MCMC object to perform Bayesian estimation of fragility functions
  by taking into account uncertainty in ground motion IM.
  '''
  
  def __init__(self, parampriors, args):
    '''
    Wrapper around Numpyro MCMC object to perform Bayesian estimation of fragility functions
    with uncertain IM values

    args:
        parampriors (dict): Prior distributions for fragility function parameters. See
                            example of default_priors above.

        args (dict): Settings used for inference (-> see the example notebooks)
                Minimally required attributes consist of:
                mcmc (dict): 
                    num_chains (int): Number of independent MCMC chains 
                    num_warmup (int): Number of warm-up steps per MCMC chain 
                    num_samples (int): Number of samples per MCMC chain
                list_bc (list): A list with all considered building classes, e.g., ['A', 'B', ...]
                list_ds (list): A list with all considered damage states, e.g., [0, 1, 2, ...]
            
                Following additional attributes are used for information purposes:
                IM (str): The considered IM, e.g., 'PGA', 'SAT0_300' 
                IM_unit (str): The unit of the considered IM, e.g., 'g [m/s2]'
                GMM (str): The used ground motion model, e.g., ChiouYoungs2014Italy 
                SCM (str): The used spatial correlation model, e.g., BodenmannEtAl2023   
    '''    
    self.priors = parampriors
    self.args = args

  def Model(self, mu, L, ds, bc):
    '''
    Prior generative model encoded in Numpyro

    args:
        mu (ArrayLike): Mean of logIM at survey sites conditional on station data
            dimension: (n_sites, )

        L (ArrayLike): Lower Cholesky decomposition of the covariance matrix of logIM
                        at survey sites conditional on station data
            dimension: (n_sites, n_sites)

        ds (ArrayLike): Observed damage states (encoded with integers)
            dimension: (n_sites,)

        bc (ArrayLike): Observed building classes (encoded with integers)
            dimension: (n_sites,)

    '''
    n_sites = ds.shape[0] # Number of survey data points

    # Prior sampling of fragility function parameters
    beta = numpyro.sample('beta', self.priors['beta']) 
    eta1 = numpyro.sample('eta1', self.priors['eta1']) 
    if 'deltas' in self.priors.keys():
        deltas = numpyro.sample('deltas', self.priors['deltas'])   

    # Prior sampling of z: dim (n_sites,)
    z = numpyro.sample('z', dist.Normal(jnp.zeros(n_sites), 1) ) 

    # Transform z to logIM at survey sites: dim (n_sites,)
    logIM = mu + jnp.matmul(L, z) 

    # Compute damage state probabilities: dim (n_sites, n_ds)
    p = get_pmf_ds_logIM(logIM, bc, eta1, deltas, beta).T 

    # Sample damage states: dim (n_sites,)
    numpyro.sample('obs', dist.Categorical(probs = p), obs = ds)

  def run_mcmc(self, mu, L, ds, bc):
    '''
    Performs MCMC using the prior generative model with inputs

    args:
        mu (ArrayLike): Mean of logIM at survey sites conditional on station data
            dimension: (n_sites, )

        L (ArrayLike): Lower Cholesky decomposition of the covariance matrix of logIM
                        at survey sites conditional on station data
            dimension: (n_sites, n_sites)

        ds (ArrayLike): Observed damage states (encoded with integers)
            dimension: (n_sites,)

        bc (ArrayLike): Observed building classes (encoded with integers)
            dimension: (n_sites,)

    '''
    if 'mcmc_seed' not in self.args.keys(): seed = 0
    else: seed = self.args['mcmc_seed']
    rng_key = jax.random.PRNGKey(seed)
    self.mcmc = MCMC(NUTS(self.Model), **self.args['mcmc'])
    self.mcmc.run(rng_key, mu = mu, L = L, 
                ds = ds, bc = bc)

  def get_posterior(self):
    return Posterior.from_mcmc(self.mcmc, args=self.args)

#---------
# Maximum likelihood inference with fixed IM values
#---------

class MLE_fixedIM(object):
    '''
    Wrapper around Numpyro SVI object to perform MLE of fragility functions
    by using fixed IM values.
    '''
    def __init__(self, init_values, args):
        '''
        Wrapper around Numpyro SVI object to perform MLE of fragility functions
        by using fixed IM values.

        args:
            parampriors (dict): Prior distributions for fragility function parameters. See
                                example of default_priors above.

            args (dict): Settings used for inference (-> see the example notebooks)
                Minimally required attributes consist of:
                mle (dict): 
                    num_iter (int): Number of iterations (default is 1000)
                list_bc (list): A list with all considered building classes, e.g., ['A', 'B', ...]
                list_ds (list): A list with all considered damage states, e.g., [0, 1, 2, ...]
            
                Following additional attributes are used for information purposes:
                IM (str): The considered IM, e.g., 'PGA', 'SAT0_300' 
                IM_unit (str): The unit of the considered IM, e.g., 'g [m/s2]'
                GMM (str): The used ground motion model, e.g., ChiouYoungs2014Italy 
                SCM (str): The used spatial correlation model, e.g., BodenmannEtAl2023         
        '''
        self.init_values = init_values
        self.optimizer = numpyro.optim.Minimize(method='BFGS')
        self.loss = Trace_ELBO()
        if 'mle' not in args.keys():
            args['mle'] = {'num_iter': 1000}
        self.args = args

    def Model(self, logIM, ds, bc):
        '''
        Probabilistic model encoded in Numpyro.

        args:
            logIM (ArrayLike): Fixed logIM values at survey sites 
                dimension: (n_sites, )

            ds (ArrayLike): Observed damage states (encoded with integers)
                dimension: (n_sites,)

            bc (ArrayLike): Observed building classes (encoded with integers)
                dimension: (n_sites,)

        '''
        # note that we need to include positive constraints;
        # in the MCMC model these constraints appear implicitly in
        # the support of the chosen prior distributions.

        eta1 = numpyro.param('eta1', init_value = self.init_values['eta1'])
        beta = numpyro.param('beta', init_value = self.init_values['beta'],
                                constraint = dist.constraints.positive)
        if 'deltas' in self.init_values.keys():
            deltas = numpyro.param('deltas', 
                                    init_value = self.init_values['deltas'],
                                    constraint = dist.constraints.positive)

        # Compute damage state probabilities: dim (n_sites, n_ds)
        p = get_pmf_ds_logIM(logIM, bc, eta1, deltas, beta).T

        # Observations: dim (n_sites,)
        numpyro.sample('obs', dist.Categorical(probs = p), obs = ds)

    def Guide(self, logIM, ds, bc):
        pass

    def run(self, logIM, ds, bc):
        '''
        Performs MLE to find fragility function parameters

        args:
            logIM (ArrayLike): Fixed logIM values at survey sites 
                dimension: (n_sites, )

            ds (ArrayLike): Observed damage states (encoded with integers)
                dimension: (n_sites,)

            bc (ArrayLike): Observed building classes (encoded with integers)
                dimension: (n_sites,)

        '''
        args_mle = self.args['mle']
        rng_key = jax.random.PRNGKey(0) # Random number generator not used
        mle = SVI(self.Model, self.Guide, self.optimizer, loss = self.loss)
        mle_result = mle.run(rng_key, num_steps = args_mle['num_iter'], 
                             logIM = logIM, ds = ds, bc = bc)
        return PointEstimates.from_dict(mle_result.params, self.args)