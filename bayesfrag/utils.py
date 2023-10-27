# ADD COPYRIGHT AND LICENSE

import jax.numpy as jnp
from jax.scipy.stats import norm
import numpyro.distributions as dist

def default_priors(n_bc: int, n_ds: int):
    '''
    Gives the prior fragility parameter distributions used in the manuscript.
    Assumes identical priors for the parameters of each building class.

    args:
        n_bc (int): Number of considered building classes

        n_ds (int): Number of considered damage states         

    returns:
        parampriors (dict): Prior parameter distributions
    '''
    parampriors = {
        'eta1':  dist.Normal(loc = jnp.ones((n_bc, 1)) * jnp.log(0.2),
                            scale = jnp.ones((n_bc, 1)) * 1.5),
        'beta': dist.InverseGamma(concentration = jnp.ones(n_bc) * 2.5,
                                rate = jnp.ones(n_bc) * 1.0 )
    }
    if n_ds > 2:
        parampriors['deltas'] = dist.Gamma(
                        concentration = jnp.ones((n_bc, n_ds-2)) * 1.5,
                        rate = jnp.ones((n_bc, n_ds-2)) * 2.5 )
    return parampriors

def get_pmf_ds_logIM(logIM, bc, eta1, deltas, beta):
    '''
    Evaluate fragility functions -> Computes damage state probabilities 
    conditional on: logIM, parameters (eta1, deltas, beta) and building classes

    args:
        logIM (ArrayLike): logarithmic intensity measure in unit g
            dimension: (n_sites,)

        bc (ArrayLike): Observed building classes (encoded with integers)
            dimension: (n_sites,)

        eta1 (ArrayLike): first threshold parameter of fragility functions 
            dimension: (n_bc, 1)            

        deltas (ArrayLike): difference between remaining threshold parameters 
            dimension: (n_bc, n_ds-2)

        beta (ArrayLike): dispersion parameters of fragility functions
            dimension: (n_bc,)

    returns:
        probs (ArrayLike): damage state probabilities
            dimension: (n_sites, n_ds)
    '''
    beta_inv = 1/beta
    # Transform deltas to threshold parameters etas
    etas = jnp.append(eta1, 
                      eta1 + jnp.cumsum(deltas, axis=1), 
                      axis=1)    
    # Evaluate fragility functions at logIM
    # Probability of reaching or exceeding a certain DS
    ccdfs = []
    ccdfs.append(jnp.ones_like(logIM))
    for ds in jnp.arange(etas.shape[1]):
      ccdfs.append(norm.cdf(logIM * beta_inv[bc] - etas[bc, ds]))
    ccdfs.append(jnp.zeros_like(logIM))
    # Compute probability of being in a certain DS
    probs = jnp.diff(1-jnp.stack(ccdfs), axis=0)
    # Add some jitter for numerical stability
    return probs + 1e-6