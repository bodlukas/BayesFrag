import numpy as np
import pandas as pd

# ['pga', 'T0_200', 'T0_300', 'T0_600']

class Sites(object):
    """Wrapper for geographic sites at which to predict ground-motion amplitudes.
    """    
    def __init__(self, coordinates, mu_logIM, tau_logIM, phi_logIM, 
                 vs30 = None, epiazimuth = None, coorunits = 'decdeg',
                 ):
        """Initialize Sites object (collection of n_sites)

        Args:
            coordinates (ArrayLike): Array with dimension (n_sites, 2)
                First column: Longitude in decimal degrees OR Easting in km or m
                Second column: Latitude in decimal degrees OR Northing in km or m

            mu_logIM (ArrayLike): Array with dimension (n_sites,)
                Mean of logIM as obtained from a GMM

            tau_logIM (ArrayLike): Array with dimension (n_sites,)
                Standard deviation of between-event residual of logIM 
                as obtained from a GMM

            phi_logIM (ArrayLike): Array with dimension (n_sites,)
                Standard deviation of within-event residual of logIM 
                as obtained from a GMM               

            vs30 (ArrayLike, optional): Vs30 values of the sites in m/s
                May be required by some spatial correlation models
            
            epiazimuth (ArrayLike, optional): Epicentral azimuth in decimal degrees
                May be required by some spatial correlation models

            coorunits (str, optional): Specify the units for the coordinates
                - 'decdeg'  : Longitude and Latitude in decimal degrees
                - 'km'      : Easting and Northing in km
                - 'm'       : Easting and Northing in m 
                Defaults to 'decdeg'           

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