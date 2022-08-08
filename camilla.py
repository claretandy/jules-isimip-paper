import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

plushist = pd.read_csv('/home/h02/hadcam/github/IllusPathwaysAR6/plushistRCMIP.csv')
plushist_long = pd.melt(plushist, id_vars=['Model', 'Scenario', 'Region', 'Variable', 'Unit', 'Mip_Era', 'Activity_Id'], var_name='Year')
rcmip_ar6 = pd.read_csv('/home/h02/hadcam/github/IllusPathwaysAR6/IllusPathsAR6_plushistRCMIP.csv')
rcmip_ar6_long = pd.melt(rcmip_ar6, id_vars=['Model', 'Scenario', 'Region', 'Variable', 'Unit'], var_name='Year')
plushist_long.drop(columns=['Mip_Era', 'Activity_Id'], inplace=True)
rcmip_alldata = pd.concat([plushist_long, rcmip_ar6_long])

sns.relplot(data=rcmip_alldata, x='Year', y='value', hue='Model')