# Copyright (c) Lukas Bodenmann, Bozidar Stojadinovic, ETH Zurich, Switzerland
# SPDX-License-Identifier: BSD-3-Clause 

import numpy as np
import warnings
from typing import Optional
from numpy.typing import ArrayLike
from .sites import Sites

EARTHRADIUS = 6371.0 # Radius of earth in km.

class SpatialCorrelationModel(object):
    """ Base class for spatial correlation models for within-event residuals 
    of ground-motion intensity measures.
    """
    def __init__(self, im_string: str) -> None:
        """
        Parameters
        ----------
        im_string : str 
            Indicate for which IM to compute correlations. 
            For PGA use 'PGA', for SA(T=0.3s) use 'SAT0_300'.
        """        

        if im_string == 'PGA':
            self.T = 0.0
        else: 
            self.T = float('.'.join( im_string[3:].split('_') ))
        
        self._required = ['coor']
        self.name = 'none'

    def check_attributes(self, sites: Sites) -> None:
        check = [getattr(sites, attr) for attr in self._required]
        if any(x is None for x in check):
            mask = [x is None for x in check]            
            raise ValueError('The correlation model ' + self.name + 
                             'requires additional site attributes ' + 
                             ', '.join(np.array(self._required)[mask]))

    def get_euclidean_distance_matrix(self, sites1: Sites, sites2: Optional[Sites] = None) -> ArrayLike: 
        """Computes Euclidean distance matrix

        Computes the distance from each site in sites1 to every other site in 
        site 1, or, if sites2 is provided, to every site in sites2.

        If coordinates are in (Longitude, Latitude) -> the method implements:
        http://williams.best.vwh.net/avform.htm#Dist , which is the same algortihm 
        used by OpenQuake (v3.16.0).

        Parameters
        ----------
        sites1 : Sites
            Primary sites for which to compute distance matrix.
        sites2 : Sites, optional
            Secondary sites. If provided, distance matrix between primary 
            and secondary sites. 

        Returns
        -------
        distances : ArrayLike
            Distance matrix in km
        """
        X1 = sites1.coor
        if sites2 is None: X2 = X1
        else: X2 = sites2.coor
        if sites1.coorunits == 'decdeg':
            lons1 = np.radians(np.reshape(X1[:, 0], [-1, 1])) 
            lats1 = np.radians(np.reshape(X1[:, 1], [-1, 1]))
            lons2 = np.radians(np.reshape(X2[:, 0], [1, -1])) 
            lats2 = np.radians(np.reshape(X2[:, 1], [1, -1]))
            dist = np.arcsin( np.sqrt( np.sin((lats1 - lats2) / 2.0) ** 2.0
                                            + np.cos(lats1) * np.cos(lats2)
                                            * np.sin((lons1 - lons2) / 2.0) ** 2.0 )
                                )
            dist = dist * (2*EARTHRADIUS)
        else:
            if sites1.coorunits == 'm': 
                X1 /= 1000 # from m to km
                X2 /= 1000 # from m to km
            sq_dist = np.sum(np.stack([
                np.square(np.reshape(X1[:,i], [-1,1]) - np.reshape(X2[:,i], [1,-1]))
                        for i in range(X1.shape[1])]), axis=0)
            sq_dist = np.clip(sq_dist, 0, np.inf)
            dist = np.sqrt(sq_dist)
        return dist
                
class EspositoIervolino2012(SpatialCorrelationModel):
    '''
    Implements model proposed in:

    - for PGA: 
        Esposito S. and Iervolino I. (2011): "PGA and PGV Spatial Correlation Models 
        Based on European Multievent Datasets"
        Bulletin of the Seismological Society of America, doi: 10.1785/0120110117

    - for SA(T): 
        Esposito S. and Iervolino I. (2012): "Spatial Correlation of Spectral 
        Acceleration in European Datas"
        Bulletin of the Seismological Society of America, doi: 10.1785/0120120068

    '''

    def __init__(self, im_string: str, dataset: str='it') -> None:
        '''
        Parameters
        ----------
        im_string : str 
            Indicate for which IM to compute correlations. 
            For PGA use 'PGA', for SA(T=0.3s) use 'SAT0_300'.
        dataset : {'it', 'esm'}, defaults to 'it'     
            Indicate which parameters to be used.  
            'it': Parameters estimated from Italian data set
            'esm': Parameters estimated from European data set
        '''
        super().__init__(im_string)
        self.corr_range = self._get_parameter_range(dataset)
        self.name = 'EspositoIervolino2012_' + dataset
        self._required = ['coor'] # Required site attributes

    def get_correlation_matrix(self, sites1: Sites, sites2: Optional[Sites] = None) -> ArrayLike:
        ''' Computes correlation matrix

        Computes the correlation coefficient from each site in sites1 to every other site in 
        site 1, or, if sites2 is provided, to every site in sites2.        

        Parameters
        ----------
        sites1 : Sites
            Primary sites for which to compute correlation matrix.
        sites2 : Sites, optional
            Secondary sites. If provided, correlation matrix between primary 
            and secondary sites. 

        Returns
        -------
        correlation : ArrayLike
            Correlation matrix
        '''
        self.check_attributes(sites1)
        if sites2 is not None: self.check_attributes(sites2)
        dist_mat = self.get_euclidean_distance_matrix(sites1, sites2)
        return np.exp(-3 * dist_mat / self.corr_range)
    
    def _get_parameter_range(self, dataset):
        if dataset == 'esm':
            if self.T == 0: corr_range = 13.5
            else: corr_range = 11.7 + 12.7 * self.T
        elif dataset == 'it': 
            if self.T == 0: corr_range = 11.5
            else: corr_range = 8.6 + 11.6 * self.T            
        return corr_range

# Parameters for Model of BodenmannEtAl
params_BodenmannEtAl = {0.01: {'LE': 16.4, 'gammaE': 0.36, 'LA': 24.9, 'LS': 171.2, 'w': 0.84},
                        0.03: {'LE': 16.9, 'gammaE': 0.36, 'LA': 25.6, 'LS': 185.6, 'w': 0.84},
                        0.06: {'LE': 16.6, 'gammaE': 0.35, 'LA': 24.4, 'LS': 190.2, 'w': 0.84},
                        0.1: {'LE': 16.3, 'gammaE': 0.34, 'LA': 23.3, 'LS': 189.8, 'w': 0.88},
                        0.3: {'LE': 15.1, 'gammaE': 0.34, 'LA': 26.1, 'LS': 199.9, 'w': 0.85},
                        0.6: {'LE': 25.6, 'gammaE': 0.37, 'LA': 24.2, 'LS': 222.8, 'w': 0.73},
                        1.0: {'LE': 29.8, 'gammaE': 0.41, 'LA': 20.5, 'LS': 169.2, 'w': 0.7},
                        3.0: {'LE': 42.1, 'gammaE': 0.46, 'LA': 18.5, 'LS': 358.0, 'w': 0.5},
                        6.0: {'LE': 70.2, 'gammaE': 0.49, 'LA': 17.3, 'LS': 372.2, 'w': 0.54}}

class BodenmannEtAl2023(SpatialCorrelationModel):
    '''
    Implements model: 
        Bodenmann L., Baker J.W. and Stojadinović B. (2023): "Accounting for path and site effects in 
        spatial ground-motion correlation models using Bayesian inference"
        Natural Hazards and Earth System Sciences, doi: 10.5194/nhess-23-2387-2023

    Parameters estimated for SA at 9 periods from PEER NGAwest2 data set using the ChiouYoungs2014 GMM.
    Note: For PGA, we take the parameters obtained for SA(T=0.01s).
    '''

    def __init__(self, im_string: str) -> None:
        '''
        Parameters
        ----------
        im_string : str 
            Indicate for which IM to compute correlations. 
            For PGA use 'PGA', for SA(T=0.3s) use 'SAT0_300'.
        '''
        super().__init__(im_string)
        self.name = 'BodenmannEtAl2023'
        self._required = ['coor', 'vs30', 'epiazi'] # Required site attributes

    def get_angular_distance_matrix(self, sites1: Sites, sites2: Optional[Sites] = None) -> ArrayLike:
        '''
        Computes matrix with differences in epicentral azimuth values of sites.
        See also doc of get_euclidean_distance_matrix.
        '''
        azimuths1 = np.radians( sites1.epiazi ).reshape(-1, 1)

        if sites2 is None:
            azimuths2 = azimuths1.T
        else:
            azimuths2 = np.radians( sites2.epiazi ).reshape(1, -1)
        cos_angle = np.cos( np.abs(azimuths1 - azimuths2) )
        distances =  np.arccos(np.clip(cos_angle, -1, 1))   
        return distances * 180/np.pi
    
    def get_soil_dissimilarity_matrix(self, sites1: Sites, sites2: Optional[Sites] = None) -> ArrayLike:
        '''
        Computes the absolute difference in vs30 values between sites.
        See also doc of get_euclidean_distance_matrix.
        '''
        vs301 = sites1.vs30.reshape(-1, 1)
        if sites2 is None:
            vs302 = vs301.T
        else:
            vs302 = sites2.vs30.reshape(1, -1)
        distances = np.abs(vs301 - vs302) 
        return distances

    def get_correlation_matrix(self, sites1: Sites, sites2: Optional[Sites] = None) -> ArrayLike:
        ''' Computes correlation matrix

        Computes the correlation coefficient from each site in sites1 to every other site in 
        site 1, or, if sites2 is provided, to every site in sites2.        

        Parameters
        ----------
        sites1 : Sites
            Primary sites for which to compute correlation matrix.
        sites2 : Sites, optional
            Secondary sites. If provided, correlation matrix between primary 
            and secondary sites. 

        Returns
        -------
        correlation : ArrayLike
            Correlation matrix
        '''
        self.check_attributes(sites1)
        if sites2 is not None: self.check_attributes(sites2)
        dist_mat_E = self.get_euclidean_distance_matrix(sites1, sites2)
        dist_mat_A = self.get_angular_distance_matrix(sites1, sites2)
        dist_mat_S = self.get_soil_dissimilarity_matrix(sites1, sites2)
        params = self._get_parameters()
        rho_E = np.exp( - np.power(dist_mat_E / params['LE'], params['gammaE']) )
        rho_A = (1 + dist_mat_A/params['LA']) * np.power(1 - dist_mat_A/180, 180/params['LA'])
        rho_S = np.exp(- dist_mat_S / params['LS'])
        return rho_E * (params['w'] * rho_A + (1-params['w']) * rho_S)
    
    def _get_parameters(self):
        if self.T in params_BodenmannEtAl.keys():
            return params_BodenmannEtAl[self.T]
        else:
            idx = (np.abs(np.asarray(self.T) - np.asarray(list(params_BodenmannEtAl.keys())))).argmin()
            Tclose = list(params_BodenmannEtAl.keys())[idx]
            warnings.warn('Use parameters for T = ' + str(Tclose) + 's instead of T = ' + str(self.T) + 's')
            return params_BodenmannEtAl[Tclose]