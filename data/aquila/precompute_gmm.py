'''
Precomputes the GMM results for the L'Aquila case study
Requires an OpenQuake environment with numpy, pandas and json.

from command line in this directory: 
'python precompute_gmm.py'

Check out Tutorial 3 for further documentation.
'''

import numpy as np
import pandas as pd
import json
import openquake.hazardlib as oq

Point = oq.geo.point.Point
PlanarSurface = oq.geo.surface.planar.PlanarSurface
BaseRupture = oq.source.rupture.BaseRupture

def get_rupture():
    '''
    The rupture metadata is from ESM: 
    https://esm-db.eu/#/event/IT-2009-0009 

    The rupture geometry is from INGV: 
    http://shakemap.ingv.it/shake4/downloadPage.html?eventid=1895389 
    The latter is identical to the ESM rupture geometry!
    '''
    f = open('rupture.json')
    rup_temp = json.load(f)
    f.close()
    rup_geom_json = rup_temp['features'][0]['geometry']
    rup_geom = np.array(rup_geom_json['coordinates'][0][0])[:-1,:]

    rupture_surface = PlanarSurface.from_corner_points(
        top_left = Point(rup_geom[0, 0], rup_geom[0, 1], rup_geom[0, 2]),
        top_right = Point(rup_geom[1, 0], rup_geom[1, 1], rup_geom[1, 2]),
        bottom_right = Point(rup_geom[2, 0], rup_geom[2, 1], rup_geom[2, 2]),
        bottom_left = Point(rup_geom[3, 0], rup_geom[3, 1], rup_geom[3, 2]),
    )
    rupture = BaseRupture(mag = 6.1, rake = -90.0, 
                        tectonic_region_type = 'Active Shallow Crust', 
                        hypocenter = Point(longitude = 13.380, 
                                            latitude = 42.342,
                                            depth = 8.3),
                        surface = rupture_surface)
    return rupture

def get_oq_args(args):
    if args['GMM'] == 'BindiEtAl2011':
        gmm = oq.gsim.bindi_2011.BindiEtAl2011()
    elif args['GMM'] == 'ChiouYoungs2014Italy':
        gmm = oq.gsim.chiou_youngs_2014.ChiouYoungs2014Italy()

    # Extract whether IM is defined for the geometric mean or RotD50
    im_definition = gmm.DEFINED_FOR_INTENSITY_MEASURE_COMPONENT.value
    if im_definition == 'Average Horizontal':
        obs_str = 'geoM_log' + args['im_string']
    elif im_definition == 'Average Horizontal (RotD50)':
        obs_str = 'rotD50_log' + args['im_string']

    if args['im_string'] == 'PGA':
        im_list = [oq.imt.PGA()]
    else:
        T = float('.'.join( args['im_string'][3:].split('_') )) 
        im_list = [oq.imt.SA(T)]
    return gmm, obs_str, im_list 

def get_epiazimuth(rupture, sites_mesh):
    lon, lat = rupture.hypocenter.longitude, rupture.hypocenter.latitude
    lons, lats = sites_mesh.lons, sites_mesh.lats    
    return oq.geo.geodetic.fast_azimuth(lon, lat, lons, lats)

def get_RuptureContext(rupture, sites_mesh, sites_vs30, 
                sites_vs30measured=None, sites_z1pt0=None):

    rctx = oq.contexts.RuptureContext()
    rctx.rjb = rupture.surface.get_joyner_boore_distance(sites_mesh)
    rctx.rrup = rupture.surface.get_min_distance(sites_mesh)
    rctx.vs30 = sites_vs30
    rctx.mag = rupture.mag * np.ones_like(rctx.rjb)
    rctx.rake = rupture.rake * np.ones_like(rctx.rjb)
    rctx.rx = rupture.surface.get_rx_distance(sites_mesh)
    rctx.ztor = rupture.surface.get_top_edge_depth() * np.ones_like(rctx.rjb)
    rctx.dip = rupture.surface.get_dip() * np.ones_like(rctx.rjb)
    if sites_z1pt0 is None:
        rctx.z1pt0 = -7.15/4 * np.log( (sites_vs30**4 + 571**4) / (1360**4 + 571**4) )
    else: 
        rctx.z1pt0 = sites_z1pt0
    if sites_vs30measured is None:
        rctx.vs30measured = False
    else:
        rctx.vs30measured = sites_vs30measured
    return rctx    

def compute_GMM(gmm, im_list, rupture_context):
    n = len(rupture_context.vs30)
    nim = len(im_list)
    mean = sigma = tau = phi = np.zeros([nim, n])
    # sigma = np.zeros([nim, n])
    # tau = np.zeros([nim, n])
    # phi = np.zeros([nim, n])
    gmm.compute(rupture_context, im_list, mean, sigma, tau, phi)
    return {'mu_logIM': mean.squeeze(), 
            'tau_logIM': tau.squeeze(), 
            'phi_logIM': phi.squeeze()}

def get_gmm_res(filepath, rupture, args):
    gmm, obs_str, im_list = get_oq_args(args)

    df = pd.read_csv(filepath)
    if filepath.split('.')[0] == 'gridmap':
        df = df[df.vs30>0].copy()

    sites_mesh = oq.geo.mesh.Mesh(df['Longitude'].values, 
                                  df['Latitude'].values, 
                                  depths=None)
    
    if 'vs30measured' not in df.columns.values:
        df['vs30measured'] = False


    # Compute GMM estimates
    rupture_context = get_RuptureContext(rupture, sites_mesh, 
                                        sites_vs30 = df['vs30'].values, 
                                        sites_vs30measured = df['vs30measured'].values)

    res = compute_GMM(gmm, im_list, rupture_context)    

    if args['GMM'] == 'ChiouYoungs2014Italy':
        res['epiazimuth'] = get_epiazimuth(rupture, sites_mesh).squeeze()

    if filepath.split('.')[0] == 'stations':
        res['obs_logIM'] = df[obs_str].values

    return res

def main():
    im_strings = ['SAT0_300', 'SAT0_300', 'PGA']
    gmms = ['BindiEtAl2011', 'ChiouYoungs2014Italy', 'BindiEtAl2011']

    rupture = get_rupture()
    for im_string, gmm in zip(im_strings, gmms):
        args = {'im_string': im_string, 'GMM': gmm}
        for file in ['survey.csv', 'stations.csv', 'gridmap.csv']:
            res = get_gmm_res(file, rupture, args)
            savefile = file.split('.')[0] + '_im_' + args['im_string'] + '_gmm_' + args['GMM'] + '.npz'
            np.savez(savefile, **res)

if __name__ == '__main__':
    main()