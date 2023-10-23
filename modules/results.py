import numpy as np
from scipy import stats
import arviz as az
import pandas as pd
import xarray as xr
# JAX-related
import jax
import jax.numpy as jnp
from scipy import linalg
from modules.utils import Sites
from modules.shakemap import GPR
#-----------
# Object to store and plot posterior results from MCMC
#-----------

class Posterior(object):
    '''
    Object to store and plot posterior results from MCMC.

    The object can be initialized from an xarray dataset 
    or directly from the numpyro mcmc object used for inference.

    '''
    def __init__(self, xarray_samples: xr.Dataset):
        """
        Initialize object directly from an xarray dataset. The latter contains 
        the results from MCMC-based inference which are saved locally in terms 
        of the sampled fragility function parameters. 

        Args:
            xarray_samples (xr.Dataset): MCMC samples from the posterior
         
        """       
        self.samples = xarray_samples.copy()
        self._compute_etas()
        # + 1 to include damage state 0 (i.e., no damage)
        self.n_ds = len(self.samples.ds.values) + 1

    @classmethod
    def from_mcmc(cls, numpyro_mcmc, args):
        """
        Initialize object directly the numpyro mcmc object used for inference.

        Args:
            numpyro_mcmc: The object 

            args (dict): Settings used for inference (-> see the example notebooks)
                Minimally required attributes consist of:
                list_bc: A list with all considered building classes, e.g., ['A', 'B', ...]
                list_ds: A list with all considered damage states, e.g., [0, 1, 2, 3, 4, 5]
                im_string: String with the considered IM, e.g., 'PGA', 'SAT0_300' 
                GMM: The used ground motion model, e.g., ChiouYoungs2014Italy 
                SCM: The used spatial correlation model, e.g., BodenmannEtAl2023            
        """       
        res_az = az.from_numpyro(numpyro_mcmc, 
                            coords = {"bc": args['list_bc'],
                                    "ds1": [args['list_ds'][1]],
                                    "ds2+": args['list_ds'][2:]},
                            dims = {"eta1": ["bc", "ds1"],
                                    "beta": ["bc"],
                                    "deltas": ["bc", "ds2+"],
                                    "z": ["sid"]})
        res_az.posterior = res_az.posterior.rename_vars({'eta1': 'eta'})
        res_az.posterior.attrs['IM'] = args['im_string']
        res_az.posterior.attrs['IM_unit'] = 'g [m/s2]'
        res_az.posterior.attrs['GMM'] = args['GMM']
        res_az.posterior.attrs['SCM'] = args['SCM']
        return cls(xarray_samples = az.extract(res_az.posterior))    

    def get_diagnostics(self):
        '''
        Computes MCMC convergence diagnostic metrics: effective sample size and
        R-hat. See also Appendix C of the Manuscript for further information.
        '''
        dfdiagnostic = az.summary(self.samples.unstack(), kind='diagnostics')
        return dfdiagnostic[['ess_bulk', 'r_hat']]

    def get_mean_fragparams(self, dataframe: bool = True, option: str = 'thetas'):
        '''
        Computes mean posterior fragility function parameters as the average
        over the posterior samples.

        Args:
            dataframe (bool): If true returns parameters as a dataframe, otherwise
                as an xarray data set. Defaults to True.

            option (str): Chosen parametrization of fragility function
                - 'thetas': The median IM which causes a structure to reach or 
                            exceed a certain DS.
                            P(DS >= ds|im) = norm.cdf( log(im/thetaDS) / beta )
                - 'etas': The threshold parameters of the cumulative probit model.
                            P(DS >= ds|im) = norm.cdf( log(im)/beta - etaDS )
        
        Note: The dispersion parameter beta is always included.
        '''
        # Compute mean over posterior samples
        meanp = self.samples[['beta', 'etas']].mean(dim=["sample"])
        # Compute mean theta
        mean_thetas = np.exp(meanp.beta.values[:,None] * meanp.etas.values)
        meanp['thetas'] = (['bc', 'ds'], mean_thetas)
        if dataframe == False:
            return meanp
        else:
            df = pd.DataFrame()
            df['bc'] = meanp.bc.values
            df['beta'] = meanp.beta.values
            for ds in meanp.ds.values:
                if option == 'thetas':
                    df['theta' + str(ds)] = meanp.sel({'ds': ds}).thetas.values
                else:
                    df['eta' + str(ds)] = meanp.sel({'ds': ds}).etas.values
            return df.set_index('bc')

    def _compute_etas(self):
        if 'deltas' in self.samples:
            etas = np.append(self.samples.eta.values, 
                self.samples.eta.values + 
                np.cumsum(self.samples.deltas.values, axis = 1),
                axis=1)
        else:
            etas = self.samples.eta.values

        self.samples['etas'] = (['bc', 'ds', 'sample'], etas)
        ds_list = np.append(self.samples['ds1'].values, self.samples['ds2+'].values)
        self.samples['etas'] = self.samples['etas'].assign_coords(
            {"ds": ds_list})

    def get_logIM_samples(self, mu_B_S, L_BB_S):
        '''
        Transforms posterior samples of z to samples of logIM. These samples
        are from the posterior distribution of logIM at the locations of 
        the surveyed buildings.

        Args:
            mu_B_S (ArrayLike): Mean of logIM at the surveyed buildings conditional
                on station data. Dimension: (n_sites,)

            L_BB_S (ArrayLike): Lower Cholesky transform of covariance matrix of logIM 
                at the surveyed buildings conditional on station data. 
                Dimension: (n_sites x n_sites)       
        '''
        if 'z' not in self.samples:
            raise ValueError('Requires access to posterior samples of z')
        return mu_B_S[:, None] + (L_BB_S @ self.samples.z.values)

    def get_mean_std_logIM(self, mu_B_S, L_BB_S):
        '''
        Computes the mean and standard deviation of the posterior logIM samples 
        at the locations of the surveyed buildings.

        Args:
            mu_B_S (ArrayLike): Mean of logIM at the surveyed buildings conditional
                on station data. Dimension: (n_sites,)

            L_BB_S (ArrayLike): Lower Cholesky transform of covariance matrix of logIM 
                at the surveyed buildings conditional on station data. 
                Dimension: (n_sites x n_sites)       
        '''
        samples_logIM = self.get_logIM_samples(mu_B_S, L_BB_S)
        return np.mean(samples_logIM, axis=1), np.std(samples_logIM, axis=1) 
  
    def plot_frag_funcs(self, ax, bc, im, color, ds_subset = None,
                        kwargsm=dict(), includeCI: bool=True, 
                        kwargsCI = {'alpha': 0.2}):
        '''
        Plot the fragility functions using the posterior samples from the Bayesian 
        estimation approach.

        Args:
            ax: Matplotlib axis object, axis to plot the fragility functions.

            bc: Building class for which to plot the fragility function.

            im: Array of IM levels for which to plot the fragility function (horizontal axis) 

            color: Matplotlib color. Uses color for all damage states and for mean and 90% CI. 

            ds_subset (Optional): If provided, only the functions for these damage states will 
                be plotted.

            kwargsm (Optional, dict): Further matplotlib attributes that control the mean function,
                e.g., {'linewidth': 1.75, 'linestyle': '--'}

            includeCI (Optional, bool): If True, 90% CI is illustrated as the difference between the
                95% and 5% quantile of all fragility function samples for each IM level.
                Defaults to True
            
            kwargsCI (Optional, dict): Further matplotlib attributes that control the 90% CI shaded area.

        '''  
        kwargsCI['color'] = kwargsm['color'] = color

        logim = np.atleast_2d(np.log(im)).T
        betas = self.samples.sel({'bc': bc}).beta.values

        if ds_subset is None: ds_list = self.samples.ds.values
        else: ds_list = ds_subset
        for i, ds in enumerate(ds_list):
            if i > 0: kwargsCI['label'] = kwargsm['label'] = None
            etas = self.samples.sel({'bc': bc, 'ds': ds}).etas.values
            if includeCI:
                ccdf = stats.norm.cdf(logim/betas.reshape(1,-1) - etas.reshape(1,-1))
                qs = np.quantile(ccdf, [0.05, 0.95], axis=1)
                ax.fill_between(im, qs[1,:], qs[0,:], **kwargsCI)
            ax.plot(im, stats.norm.cdf(logim/betas.mean() - etas.mean()), 
                    **kwargsm)
            
    def save_as_netcdf(self, filepath: str, include_z: bool = False, **kwargs):
        '''
        Save posterior samples to disk using the netcdf format and the xarray package.
        By default, only the fragility function parameter samples are included.

        Args:
            filepath (str): Path to which to save this dataset.

            include_z (bool): Whether to include posterior samples of z. If True, the file size
                of the local copy can be large. To alleviate this, one can first compute the 
                desired IM statistics and store these statistics separately 
                (-> see the notebook examples).
                Defaults to False

            kwargs (dict): Further arguments passed to xarray.to_netcdf().

        '''  
        vars = ['beta', 'eta', 'deltas']
        if include_z: vars = vars + ['z']
        self.samples[vars].unstack().to_netcdf(filepath, **kwargs)
      
#-----------
# Object to compute and sample from the posterior predictive distribution of IMs
#-----------

class PosteriorPredictiveIM(object):
    '''
    Object to compute and sample from the posterior predictive distribution of IMs
    at other sites than the building survey sites.
    '''
    def __init__(self, GPR: GPR, survey_sites: Sites, jitter=1e-5):
        '''
        Args:
            GPR (GPR): Shake map computer

            survey_sites (Sites): Surveyed building sites
        '''
        self.gpr = GPR
        self.survey_sites = survey_sites
        self.jitter = jitter
        self._precompute()
        
    
    def sample(self, seed, target_sites, z_samples, L_BB_S, full_cov = False):
        '''
        Sample logIM values from the posterior predictive at the target_sites
        For each sample from the posterior of logIM at survey sites:
            - compute the posterior predictive at target sites
            - generate one sample from the posterior predictive
        
        The workflow is further explained in the example notebook.

        Args:
            seed (): seed for JAX random number generator

            target_sites (Sites): Sites at which we draw samples from the posterior predictive

            z_samples (ArrayLike): Posterior samples of z
                dimension: (n_samples, n_survey_sites)

            L_BB_S (ArrayLike): Lower Cholesky Decomposition of Sigma_BB_S

            full_cov (bool): if True, generate samples from the multivariate posterior predictive
                            if False, generate samples from the univariate posterior predictive

        Returns:
            sam_logIM (ArrayLike): Samples from the posterior predictive IM at the target_sites
                dimension: (n_samples, n_target_sites)
        '''
        # Covariance matrix between prior logIM at target and survey sites
        Sigma_TB = self.gpr.getCov(target_sites, self.survey_sites)
        # Covariance matrix between prior logIM at station and target sites
        Sigma_ST = self.gpr.getCov(self.gpr.sites, target_sites)
        # Mean and covariance matrix of logIM at target sites conditional on station data
        mu_T_S, Sigma_TT_S = self.gpr.predict(target_sites, full_cov = full_cov)

        sam_logIM = self._sampler(seed, z_samples, Sigma_TB, Sigma_ST, 
                               mu_T_S, Sigma_TT_S, L_BB_S, full_cov=full_cov)
        return sam_logIM

    def _precompute(self):
        Sigma_SS = self.gpr.getCov(self.gpr.sites)
        Sigma_SS = Sigma_SS + np.eye(self.gpr.sites.n_sites)*self.jitter
        self.L_SS = np.linalg.cholesky(Sigma_SS)
        # (L_SS)^-1 Sigma_SB
        self.A_SB = linalg.solve_triangular(self.L_SS,
                    self.gpr.getCov(self.gpr.sites, self.survey_sites),
                    lower=True)

    def _sampler(self, seed, z_samples, Sigma_TB, Sigma_ST, 
                               mu_T_S, Sigma_TT_S, L_BB_S, full_cov):
        rng_key = jax.random.PRNGKey(seed)
        num_samples = z_samples.shape[1]
        # (L_SS)^-1 Sigma_ST
        A_ST = jax.scipy.linalg.solve_triangular(self.L_SS, Sigma_ST, lower=True)
        # Covariance matrix between logIM at target and survey sites conditional on station data
        Sigma_TB_S = Sigma_TB - (A_ST.T @ self.A_SB)
        # (L_BB_S)^-1 Sigma_BT_S
        A_BT_S = jax.scipy.linalg.solve_triangular(L_BB_S, Sigma_TB_S.T, lower=True)
        # Mean of logIM at target sites conditional on posterior samples z 
        mu_T_S_DS = mu_T_S + (A_BT_S.T @ (z_samples)).T                 
        if full_cov:
            # Covariance matrix of logIM at target sites conditional on station and damage data
            Sigma_TT_S_DS = Sigma_TT_S - A_BT_S.T @ A_BT_S
            Sigma_TT_S_DS = jnp.clip(Sigma_TT_S_DS, a_min = 0)
            sam_logIM = (mu_T_S_DS + jax.random.multivariate_normal(rng_key, jnp.zeros_like(mu_T_S), 
                                                          Sigma_TT_S_DS, shape=(num_samples,)))
        else:
            # Variance of logIM at target sites conditional on station and damage data
            var_T_S_DS = jnp.clip(Sigma_TT_S - jnp.sum(A_BT_S**2, axis=0), a_min=0)
            sigma_T_S_DS = jnp.sqrt(var_T_S_DS)
            sam_logIM = (mu_T_S_DS + sigma_T_S_DS * jax.random.normal(rng_key, shape=mu_T_S_DS.shape))
        return sam_logIM

#-----------
# Object to store and plot point estimates of fragility parameters
#-----------

class PointEstimates(object):
    '''
    Object to store and plot posterior results from MCMC.

    The object can be initialized from an xarray dataset 
    or directly from a dictionary with the inferred parameters.

    '''
    def __init__(self, xarray_params):
        """
        Initialize object directly from an xarray dataset. The latter contains 
        the point estimates of the fragility function parameters. 

        Args:
            xarray_params (xr.Dataset): Point estimates of fragility parameters.
         
        """    
        self.params = xarray_params
        self._compute_etas()
        # + 1 to include damage state 0 (i.e., no damage)
        self.n_ds = len(self.params.ds.values) + 1


    @classmethod
    def from_dict(cls, res_dict, args):
        """
        Initialize object directly from a dictionary with the inferred parameters.

        Args:
            res_dict: Dictionary with the inferred parameters ['beta', 'eta1', 'deltas']

            args (dict): Settings used for inference (-> see the example notebooks)
                Minimally required attributes consist of:
                list_bc: A list with all considered building classes, e.g., ['A', 'B', ...]
                list_ds: A list with all considered damage states, e.g., [0, 1, 2, 3, 4, 5]
                im_string: String with the considered IM, e.g., 'PGA', 'SAT0_300' 
                GMM: The used ground motion model, e.g., ChiouYoungs2014Italy 
                SCM: The used spatial correlation model, e.g., BodenmannEtAl2023            
        """   
        betas = xr.DataArray(res_dict['beta'], dims = ['bc'], 
                    coords = [args['list_bc']], 
                    name = 'beta')
        ds = xr.Dataset({'beta': betas})
        for key in res_dict.keys():
            if key == 'beta': continue
            if key == 'eta1': dim_ds = 'ds1'; coor_ds = [args['list_ds'][1]]
            elif key == 'deltas': dim_ds = 'ds2+'; coor_ds = args['list_ds'][2:]
            else: dim_ds = 'ds'; coor_ds = args['list_ds'][1:]
            ds[key] = xr.DataArray(res_dict[key], dims = ['bc', dim_ds], 
                                coords = [args['list_bc'], coor_ds], 
                                name = key)
        ds = ds.rename_vars({'eta1': 'eta'})
        ds.attrs['IM'] = args['im_string']
        ds.attrs['IM_unit'] = 'g [m/s2]'
        ds.attrs['GMM'] = args['GMM']
        ds.attrs['SCM'] = args['SCM']
        return cls(xarray_params = ds)    

    def get_fragparams(self, dataframe = True, option = 'thetas'):
        '''
        Collects estimated fragility function parameters.

        Args:
            dataframe (bool): If true returns parameters as a dataframe, otherwise
                as an xarray data set. Defaults to True.

            option (str): Chosen parametrization of fragility function
                - 'thetas': The median IM which causes a structure to reach or 
                            exceed a certain DS.
                            P(DS >= ds|im) = norm.cdf( log(im/thetaDS) / beta )
                - 'etas': The threshold parameters of the cumulative probit model.
                            P(DS >= ds|im) = norm.cdf( log(im)/beta - etaDS )
        
        Note: The dispersion parameter beta is always included.
        '''
        if 'etas' not in self.params:
            self._compute_etas()
        if 'thetas' not in self.params:
            thetas = np.exp(self.params.beta.values[:,None] * self.params.etas.values)
            self.params['thetas'] = (['bc', 'ds'], thetas)
        if dataframe == False:
            return self.params[['beta', option]]
        else:
            df = pd.DataFrame()
            df['bc'] = self.params.bc.values
            df['beta'] = self.params.beta.values
            for ds in self.params.ds.values:
                if option == 'thetas':
                    df['theta' + str(ds)] = self.params.sel({'ds': ds}).thetas.values
                else:
                    df['eta' + str(ds)] = self.params.sel({'ds': ds}).etas.values
        return df.set_index('bc')

    def _compute_etas(self):
        if 'thetas' not in self.params:
            if 'ds2+' in self.params:
                etas = np.append(self.params.eta.values, 
                    self.params.eta.values + 
                    np.cumsum(self.params.deltas.values, axis = 1),
                    axis=1)
                if 'ds' in self.params: coords = self.params.ds.values
                else: coords = np.append(self.params['ds1'].values, 
                                         self.params['ds2+'].values)
            else:
                etas = self.params.eta.values
                coords = self.params['ds1'].values
        else: 
            etas = np.log(self.params['thetas'].values) / self.params.beta.values[:,None]
            coords = self.params.ds.values
        self.params['etas'] = (['bc', 'ds'], etas)
        self.params['etas'] = self.params['etas'].assign_coords(
            {"ds": coords})
  
    def plot_frag_funcs(self, ax, bc, im, color = None, kwargs=dict()):
        '''
        Plot the fragility functions using the posterior samples from the Bayesian 
        estimation approach.

        Args:
            ax: Matplotlib axis object, axis to plot the fragility functions.

            bc: Building class for which to plot the fragility function.

            im: Array of IM levels for which to plot the fragility function (horizontal axis) 

            color: Matplotlib color. Uses color for all damage states and for mean and 90% CI. 

            ds_subset (Optional): If provided, only the functions for these damage states will 
                be plotted.

            kwargs (Optional, dict): Further matplotlib attributes that control the mean function,
                e.g., {'linewidth': 1.75, 'linestyle': '--'}

        '''  
        kwargs['color'] = color
        betas = self.params.sel({'bc': bc}).beta.values
        for i, ds in enumerate(self.args['list_ds'][1:]):
            if i > 0: kwargs['label'] = None
            etas = self.params.sel({'bc': bc, 'ds': ds}).etas.values
            ax.plot(im, stats.norm.cdf(np.log(im)/betas - etas), 
                    **kwargs)
            
    def save_as_netcdf(self, filepath: str, **kwargs):
        '''
        Save estimated parameters to disk using the netcdf format and the xarray package.

        Args:
            filepath (str): Path to which to save this dataset.

            kwargs (dict): Further arguments passed to xarray.to_netcdf().

        '''  
        vars = ['beta', 'eta', 'deltas']
        self.params[vars].to_netcdf(filepath, **kwargs)