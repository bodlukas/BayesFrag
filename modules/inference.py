# JAX-related
import jax.numpy as jnp
from jax.scipy.stats import norm

# Numpyro-related
import numpyro
import numpyro.distributions as dist

def get_pmf_ds_logIM(logIM, etas, beta, bc, n_ds):
    '''
    Evaluate fragility functions -> Computes damage state probabilities 
    conditional on: logIM, parameters (etas, beta) and building classes

    args:
        logIM (ArrayLike): logarithmic intensity measure in unit g
            dimension: (n_sites,)

        etas (ArrayLike): threshold parameters of fragility functions 
            dimension: (n_bc, n_ds-1)

        beta (ArrayLike): dispersion parameters of fragility functions
            dimension: (n_bc,)

        bc (ArrayLike): Observed building classes (encoded with integers)
            dimension: (n_sites,)

        n_ds (int): Number of possible damage states

    returns:
        probs (ArrayLike): damage state probabilities
            dimension: (n_sites, n_ds)
    '''
    beta_inv = 1/beta
    # Evaluate fragility functions at logIM
    # Probability of reaching or exceeding a certain DS
    ccdfs = []
    ccdfs.append(jnp.ones_like(logIM))
    for ds in jnp.arange(n_ds-1):
      ccdfs.append(norm.cdf(logIM * beta_inv[bc] - etas[bc, ds]))
    ccdfs.append(jnp.zeros_like(logIM))
    # Compute probability of being in a certain DS
    probs = jnp.diff(1-jnp.stack(ccdfs), axis=0)
    # Add some jitter for numerical stability
    return probs + 1e-6

def Model_MCMC(mu, L, ds, bc, n_ds, n_bc):
    '''
    Prior generative model encoded in Numpyro

    args:
        mu (ArrayLike): Mean of logIM at survey sites conditional on station data

        L (ArrayLike): Lower Cholesky decomposition of the covariance matrix of logIM
                        at survey sites conditional on station data

        ds (ArrayLike): Observed damage states (encoded with integers)

        bc (ArrayLike): Observed building classes (encoded with integers)

        n_ds (int): Number of possible damage states

        n_bc (int): Number of possible building classes
    '''

    N = ds.shape[0] # Number of data points

    # Prior sampling distributions for fragility function parameters
    eta1 = numpyro.sample('eta1',
                          dist.Normal(jnp.ones((n_bc, 1)) * np.log(0.2),
                                      jnp.ones((n_bc, 1)) * 1.5) )

    deltas = numpyro.sample('deltas',
                          dist.Gamma(concentration = jnp.ones((n_bc, n_ds-2)) * 1.5,
                                             rate = jnp.ones((n_bc, n_ds-2)) * 2.5 ) )

    beta = numpyro.sample('beta',
                          dist.InverseGamma(concentration = jnp.ones(n_bc) * 2.5,
                                                    rate = jnp.ones(n_bc) * 1.0 ) )
    
    # Transform deltas to threshold parameters etas
    etas = jnp.append(eta1, eta1 +
                           jnp.cumsum(deltas, axis=1), axis=1)

    # Prior sampling distribution of z
    z = numpyro.sample('z', dist.Normal(jnp.zeros(N), 1) )

    # Transform z to logIM at survey sites
    logIM = mu + jnp.matmul(L, z)

    # Compute damage state probabilities
    p = get_pmf_ds_logIM(logIM, etas, beta, bc, n_ds).T

    # Likelihood
    numpyro.sample('obs', dist.Categorical(probs = p), obs = ds)