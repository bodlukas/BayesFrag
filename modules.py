#%%
import pandas as pd
import numpy as np
# %%
df = pd.read_csv('data_onedim/OneDim_buildingso.csv')
# %%
df['VulClass'] = 0
df['rec_logPGA'] = np.log(df.pga.values)
df['obs_DS'] = df.DS.values

# %%
df[['x', 'y', 'vs30', 'mu_logPGA', 'tau_logPGA', 'phi_logPGA', 
    'rec_logPGA', 'VulClass', 'obs_DS']].to_csv('data_onedim/OneDim_buildings.csv', index=False)
# %%
