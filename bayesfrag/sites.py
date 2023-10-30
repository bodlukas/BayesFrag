# Copyright (c) Lukas Bodenmann, Bozidar Stojadinovic, ETH Zurich, Switzerland
# SPDX-License-Identifier: BSD-3-Clause 

import numpy as np
import pandas as pd
from typing import Optional
from numpy.typing import ArrayLike

class Sites(object):
    """Wrapper for geographic sites at which to predict ground-motion amplitudes.
    """    
    def __init__(self, coordinates: ArrayLike, mu_logIM: ArrayLike, tau_logIM: ArrayLike, 
                phi_logIM: ArrayLike, vs30: Optional[ArrayLike] = None, 
                epiazimuth: Optional[ArrayLike] = None, coorunits: str = 'decdeg',
                 ) -> None:
        """Initialize Sites object (collection of n_sites)

        Parameters
        ----------
        coordinates : ArrayLike, dimension (n_sites, 2)
            First column: Longitude in decimal degrees OR Easting in km or m
            Second column: Latitude in decimal degrees OR Northing in km or m
        mu_logIM : ArrayLike, dimension (n_sites,)
            Mean of logIM as obtained from a GMM
        tau_logIM : ArrayLike, dimension (n_sites,)
            Standard deviation of between-event residual of logIM as obtained from a GMM        
        phi_logIM : ArrayLike, dimension (n_sites,)
            Standard deviation of within-event residual of logIM as obtained from a GMM
        vs30 : ArrayLike, optional, dimension (n_sites,)
            Vs30 values of the sites in m/s
        epiazimuth : ArrayLike, optional, dimension (n_sites,)
            Epicentral azimuth in decimal degrees   
        coorunits : {'decdeg', 'km', 'm'}, optional, defaults to 'decdeg'
            Specify the units for the coordinates
            - 'decdeg'  : Longitude and Latitude in decimal degrees
            - 'km'      : Easting and Northing in km
            - 'm'       : Easting and Northing in m                     
        """        
        assert coordinates.shape[1] == 2, 'Coordinates should have dimension (n_sites, 2)'
        self.coor = coordinates
        self.n_sites = self.coor.shape[0]
        for var in [mu_logIM, tau_logIM, phi_logIM, vs30, epiazimuth]:
            if var is not None:
                assert var.shape[0] == self.n_sites, var + ' should have length (n_sites,)'        
        self.mu_logIM = mu_logIM
        self.tau_logIM = tau_logIM
        self.phi_logIM = phi_logIM
        self.vs30 = vs30
        self.epiazi = epiazimuth
        self.coorunits = coorunits

    @classmethod
    def from_df(cls, df: pd.DataFrame, column_mapping: dict, coorunits: str = 'decdeg') -> None:
        """Initialize Sites object (collection of n_sites) from a pandas dataframe

        Parameters
        ----------
        df : Pandas dataframe, dimension (n_sites, 2)
            Dataframe with length n_sites and site attributes specified in the columns
            of column_mapping.
        column_mapping: dict, 
            Dictionary with a mapping between site attributes (see above)
            and column names in the data frame, e.g., 
            {'coordinates': ['Longitude, 'Latitude'],
                'mu_logIM': 'mean_logPGA', ...}
        coorunits : {'decdeg', 'km', 'm'}, optional, defaults to 'decdeg'
            Specify the units for the coordinates
            - 'decdeg'  : Longitude and Latitude in decimal degrees
            - 'km'      : Easting and Northing in km
            - 'm'       : Easting and Northing in m      

        """
        params_new = {}
        for key, value in column_mapping.items():
            params_new[key] = df[value].values

        if 'vs30' not in column_mapping.keys():
            params_new['vs30'] = None
        if 'epiazimuth' not in column_mapping.keys():
            params_new['epiazimuth'] = None      
  
        params_new['coorunits'] = coorunits

        return cls(**params_new)
