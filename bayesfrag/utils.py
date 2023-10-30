# ADD COPYRIGHT AND LICENSE

import jax.numpy as jnp
from jax.scipy.stats import norm
import numpyro.distributions as dist

from jax.typing import ArrayLike
from jax import Array

def default_priors(n_bc: int, n_ds: int) -> dict:
    '''
    Gives the prior fragility parameter distributions used in the manuscript.
    Assumes identical priors for the parameters of each building class.

    Parameters
    ----------
    n_bc : int
        Number of considered building classes
    n_ds : int
        Number of considered damage states

    Returns
    ----------
    parampriors : dict
        Prior numpyro distributions for parameters 'eta1', 'beta', 
        and, if more than two damage states, 'deltas'.
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

def get_pmf_ds_logIM(logIM: ArrayLike, bc: ArrayLike, beta: ArrayLike, 
                    eta1: ArrayLike, deltas: ArrayLike) -> Array:
    '''
    Evaluate fragility functions -> Computes damage state probabilities 
    conditional on: logIM, parameters (beta, eta1, deltas) and building classes.

    This is intended to be applied within Numpyro-based inference and 
    implemented in jax.numpy for JIT-compilation.

    Parameters
    ----------
    logIM : ArrayLike, dimension (n_sites, )
        Logarithmic intensity measure in unit g
    bc : ArrayLike, dimension (n_sites, )
        Observed building classes (encoded with integers)
    beta : ArrayLike, dimension (n_bc, )
        Dispersion parameters of fragility functions
    eta1 : ArrayLike, dimension (n_bc, 1)
        First threshold parameter of fragility functions
    deltas : ArrayLike, dimension (n_bc, n_ds-2)
        Difference between remaining threshold parameters
    
    Returns
    ----------
    probs : ArrayLike, dimension (n_sites, n_ds)
        Damage state probabilities
    '''
    beta_inv = 1/beta
    # Transform deltas to threshold parameters etas
    if deltas is not None:
        etas = jnp.append(eta1, 
                        eta1 + jnp.cumsum(deltas, axis=1), 
                        axis=1)    
    else:
        etas = eta1
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