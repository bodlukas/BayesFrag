'''
This script reproduces the results for the L'Aquila case
study presented in the manuscript. 

from command line in this directory: 
'python main_aquila_casestudy.py'

Note that this script is computationally expensive. We
performed the computations on a NVIDIA V100 tensor core 
GPU with 25 GB of RAM, and parallelized various
for-loops.

Check out the tutorials for a thorough explanation of the 
estimation and post-processing computations.
'''

import os
import numpyro
import pandas as pd
import numpy as np
from bayesfrag.spatialcorrelation import EspositoIervolino2012, BodenmannEtAl2023
from bayesfrag.sites import Sites
from bayesfrag.gpr import GPR
from bayesfrag.utils import default_priors
from bayesfrag.inference import Bayesian_MCMC, MLE_fixedIM
from bayesfrag.postprocess import PosteriorPredictiveIM

numpyro.enable_x64()

def get_args(im_string, gmmcombo, mcmc_subsampling_seed=None):
    args = {
        'IM': im_string,
        'IM_unit': 'g [m/s2]',
        'list_ds': [0, 1, 2, 3, 4, 5], 
        'list_bc': ['A-L', 'A-MH', 'B-L', 'B-MH', 'C1-L', 'C1-MH'], 
        'mcmc_seed': 0, # Seed for MCMC
        'mcmc': { # Check numpyro.infer.MCMC for valid settings!
            'num_samples': 750, # Number of samples per Markov chain
            'num_warmup': 1000, # Number of warmup steps per Markov chain
            'num_chains': 4 # Number of Markov chains    
                },
        'seed_subsampling': mcmc_subsampling_seed,
        'subsamples': 4, # 25% of the data set used for MCMC !!
        'subsample_number': 1,
        'gmmcombo': gmmcombo
    }
    if gmmcombo == 1:
        args['GMM'] = 'BindiEtAl2011'
        args['SCM'] = 'EspositoIervolino2012'
    elif gmmcombo == 2:
        args['GMM'] = 'ChiouYoungs2014Italy'
        args['SCM'] = 'BodenmannEtAl2023'      
    return args

def get_column_site_mapping(args):
    mapping = {
        'coordinates': ['Longitude', 'Latitude'],
        'mu_logIM': 'mu_logIM',
        'tau_logIM': 'tau_logIM',
        'phi_logIM': 'phi_logIM'
        }
    if args['SCM'] == 'BodenmannEtAl2023':
        mapping['vs30'] = 'vs30'
        mapping['epiazimuth'] = 'epiazimuth'
    return mapping

def stratified_split(df, seed_subsample, nfolds, fold_number):
    rng = np.random.default_rng(seed = seed_subsample)
    frac = 1/nfolds
    dfs = []
    for i in range(nfolds-1):
        frac = 1/nfolds / (1-i*1/nfolds)
        grouped = df.groupby(['BuildingClass', 'DamageState'])
        dft = grouped.sample(frac=frac, random_state=rng)
        dfs.append(dft)
        df = df.drop(index=dft.index)
    dfs.append(df)
    return dfs[fold_number-1]

def import_data(file, path_data, args, subsample = False):
    column_site_mapping = get_column_site_mapping(args)
    df = pd.read_csv(path_data + file, index_col = 'id')
    file_gmm = file.split('.')[0] + '_im_' + args['IM'] + '_gmm_' + args['GMM'] + '.npz'
    res_gmm = np.load(path_data + file_gmm)
    for key, value in res_gmm.items():
        df[key] = value

    if subsample: # Generate subsamples of survey data for MCMC
        df = stratified_split(df, args['seed_subsampling'], args['subsamples'], args['subsample_number'])

    sites = Sites.from_df(df, column_site_mapping)

    if file.split('.')[0] == 'stations':
        data = {'sites': sites, 'obs_logIM': df['obs_logIM'].values}
    elif file.split('.')[0] == 'survey':
        obs_BC = pd.Categorical(df.BuildingClass.values, categories = args['list_bc'], 
                                ordered=True).codes
        obs_DS = pd.Categorical(df.DamageState.values, categories = args['list_ds'], 
                                ordered = True).codes    
        data = {'sites': sites, 'obs_DS': obs_DS, 'obs_BC': obs_BC}    
    return data

def import_mapdata(file, path_data, args, batchsize = 4000):
    column_site_mapping = get_column_site_mapping(args)
    df = pd.read_csv(path_data + file, index_col = 'id')
    df = df[df.vs30>0]
    file_gmm = file.split('.')[0] + '_im_' + args['IM'] + '_gmm_' + args['GMM'] + '.npz'
    res_gmm = np.load(path_data + file_gmm)
    for key, value in res_gmm.items():
        df[key] = value

    if batchsize is not None:
        nfolds = int(np.floor(len(df)/batchsize))
        dfs = np.split(df,  [i*batchsize for i in np.arange(nfolds)], axis=0)
        sites = [Sites.from_df(dfi, column_site_mapping) for dfi in dfs[1:]]
        return sites
    else:
        return Sites.from_df(df, column_site_mapping)

def condition_on_station_data(path_data, args):
    station_data = import_data('stations.csv', path_data, args)
    if args['SCM'] == 'EspositoIervolino2012':
        scm = EspositoIervolino2012(args['IM'])
    elif args['SCM'] == 'BodenmannEtAl2023':
        scm = BodenmannEtAl2023(args['IM'])
    # Computations
    gpr = GPR(SCM=scm) # Initialize
    gpr.fit(station_data['sites'], station_data['obs_logIM'], jitter=1e-4) # Add station data
    return gpr

def perform_mcmc(gpr, survey_data, args, return_L = False):
    mu_B_S, Sigma_BB_S = gpr.predict(survey_data['sites']) # Compute parameters
    L_BB_S = np.linalg.cholesky(Sigma_BB_S) # Lower Cholesky transform
    del Sigma_BB_S
    parampriors = default_priors(n_bc = len(args['list_bc']), 
                                n_ds = len(args['list_ds']))    
    # Initialize
    bayes_mcmc = Bayesian_MCMC(parampriors, args)

    # Perform MCMC
    bayes_mcmc.run_mcmc(mu = mu_B_S, L = L_BB_S, 
            ds = survey_data['obs_DS'], bc = survey_data['obs_BC'])
    
    # Collect results in a Posterior object
    posterior = bayes_mcmc.get_posterior()
    if return_L:
        return posterior, L_BB_S
    else:
        return posterior

def main_mcmc(path_data, path_res, args, gpr, save = True, predict_map = False):

    survey_data = import_data('survey.csv', path_data, args, subsample = True)
    posterior, L_BB_S = perform_mcmc(gpr, survey_data, args, return_L = True)

    if save:
        savestr = ('Bayesian_frag_Aquila_im' + args['IM'] + '_GMMcombo' + str(args['gmmcombo']) +                     
                    '_seedsubsampling' + str(args['seed_subsampling']) + '.nc')
        posterior.save_as_netcdf(path_res + savestr)

    if predict_map:
        map_sites = import_mapdata('gridmap.csv', path_data, args)
        postpredIM = PosteriorPredictiveIM(GPR = gpr, survey_sites = survey_data['sites'])
        mean_logIM = []; std_logIM = []
        for sites in map_sites:
            samples = postpredIM.sample(args['mcmc_seed'], sites, posterior.samples['z'].values, 
                                        L_BB_S, full_cov = False)
            mean_logIM.append(np.mean(samples, axis=0))
            std_logIM.append(np.std(samples, axis=0))
        resIM = {'mean_logIM': np.hstack(mean_logIM), 'std_logIM': np.hstack(std_logIM)}
        del mean_logIM, std_logIM, map_sites, postpredIM, L_BB_S, survey_data
        if save:
            savestr = ('PosteriorIM_Aquila_im' + args['IM'] + '_GMMcombo' + str(args['gmmcombo']) + 
                            '_seedsubsampling' + str(args['seed_subsampling']) + '.nc')
            np.savez(path_res + savestr, **resIM)
        else:
            return posterior, resIM
    else:
        if save == False:
            return posterior

def perform_mle(gpr, survey_data, args):
    mu_B_S, _ = gpr.predict(survey_data['sites'], full_cov = False)

    parampriors = default_priors(n_bc = len(args['list_bc']), 
                                n_ds = len(args['list_ds'])) 
    # Specify initial values for parameters
    init_values = dict()
    for param in parampriors.keys():
        init_values[param] = parampriors[param].mean

    mle_fixedIM = MLE_fixedIM(init_values, args)
    res_fixedIM = mle_fixedIM.run(logIM = mu_B_S, ds = survey_data['obs_DS'], 
                                bc = survey_data['obs_BC'])
    return res_fixedIM

def main_mle(path_data, path_res, args, gpr, save = True, predict_map = False):
    survey_data = import_data('survey.csv', path_data, args, subsample=False)
    params_fixedIM = perform_mle(gpr, survey_data, args)

    if save:
        savestr = ('FixedIM_frag_Aquila_im' + args['IM'] + '_GMMcombo' + str(args['gmmcombo']) + '.nc')
        params_fixedIM.save_as_netcdf(path_res + savestr)

    if predict_map:
        map_sites = import_mapdata('gridmap.csv', path_data, args, batchsize=None)
        mean_logIM, var_logIM = gpr.predict(map_sites, full_cov = False)
        resIM = {'mean_logIM': mean_logIM, 'std_logIM': np.sqrt(var_logIM)}
        del mean_logIM, var_logIM, map_sites
        if save:
            savestr = ('PriorIM_Aquila_im' + args['IM'] + '_GMMcombo' + str(args['gmmcombo']) + '.nc')
            np.savez(path_res + savestr, **resIM)
        else: 
            return params_fixedIM, resIM
    else:
        if save == False:
            return params_fixedIM  

if __name__ == '__main__':
    save = True
    path_data = os.path.join('data', 'aquila', '')
    path_res = os.path.join('results', 'aquila', '')
    im_strings = ['SAT0_300', 'PGA']
    gmmcombos = [1, 2]
    subsampling_seed = 31
    # Perform estimations for two GMM combinations and two IMs
    for im_string in im_strings:
        if im_string == 'SAT0_300': 
            predict_map = True
            gmmcombos = [1, 2]
        else: 
            predict_map = False
            gmmcombos = [1]

        for gmmcombo in gmmcombos:
            args = get_args(im_string, gmmcombo, subsampling_seed)
            gpr = condition_on_station_data(path_data, args)
            main_mle(path_data, path_res, args, gpr, save, predict_map)
            main_mcmc(path_data, path_res, args, gpr, save, predict_map)
            del gpr

    # Perform MCMC with three additional data sub-samples
    im_string = 'SAT0_300'
    gmmcombo = 1 
    predict_map = False
    for subsampling_seed in [11, 21, 41]:
        args = get_args(im_string, gmmcombo, subsampling_seed)
        gpr = condition_on_station_data(path_data, args)
        main_mcmc(path_data, path_res, args, gpr, save, predict_map)
        del gpr