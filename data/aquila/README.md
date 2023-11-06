**This folder contains:**

The pre-processed damage survey data, `survey.csv`, from the 2009 L'Aquila earthquake event. The raw data set of inspection results is available from [Da.D.O.](https://egeos.eucentre.it/danno_osservato/web/danno_osservato?lang=EN). The raw data is described in Dolce et al. (2019) and the pre-processing steps are described in Bodenmann et al. (2023). 

The seismic network station data, `stations.csv`, and rupture characteristics, `rupture.json`, from the 2009 L'Aquila earthquake event as obtained from the Engineering Strong Motion database (Luzi et al., 2020).

Gridded sites to visualize maps of prior and posterior ground motion intensity estimates, `gridmap.csv`. These gridded sites are obtained from Mori et al. (2020). 

The script `precompute_gmm.py` computes the GMM estimates at the sites of surveyed buildings, seismic stations and gridded map locations using the OpenQuake engine. See Tutorial 3 for further explanations on this pre-processing step. 

**References**

Bodenmann, L., Baker, J., and Stojadinovic B. (2023): Accounting for ground motion uncertainty in empirical seismic fragility modeling. Engineering Archive [preprint], [doi:10.31224/3336](https://doi.org/10.31224/3336)

Dolce, M., Speranza, E., Giordano, F., et al. (2019): Observed damage database of past Italian earthquakes: The Da.D.O. WebGIS. Bollettino di Geofisica Teorica ed Applicata, [doi:10.4430/bgta0254](https://doi.org/10.4430/bgta0254)

Luzi, L., Lanzano, G., Felicetta, C., et al. (2020): Engineering Strong Motion Database (ESM) (Version 2.0), [doi:10.13127/ESM.2](https://doi.org/10.13127/ESM.2)

Mori, F., Mendicelli, A., Moscatelli, M., et al. (2020): Data for: A new Vs30 map for Italy based on the seismic microzonation dataset, Mendeley Data, V1, [doi:10.17632/8458tgzc73.1](https://doi.org/10.17632/8458tgzc73.1)