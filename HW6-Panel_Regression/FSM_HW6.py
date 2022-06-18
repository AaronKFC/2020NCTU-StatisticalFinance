# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 19:12:30 2020

@author: AllenPC
"""
import pandas as pd
import statsmodels.api as sm
from linearmodels.panel import PanelOLS

df = pd.read_csv('dairy.csv')
df = df.set_index(["FARM","YEAR"])


'''##### Q1: Pooled Regression #####'''
pool_regressor = PanelOLS.from_formula("YIT ~ 1 + X1 + X2 + X3 + X4", data=df)                          
print(pool_regressor.fit())


'''##### Q2: Panel Regression fixed over time #####'''
panel_regressor_Tfe = PanelOLS.from_formula("YIT ~ 1 + X1 + X2 + X3 + X4 + TimeEffects", data=df)                          
print(panel_regressor_Tfe.fit())


'''##### Q3: Panel Regression fixed over farms #####'''
panel_regressor_Efe = PanelOLS.from_formula("YIT ~ 1 + X1 + X2 + X3 + X4 + EntityEffects", data=df)                          
print(panel_regressor_Efe.fit())


'''##### Q4: Panel Regression with both time and farms effect #####'''
panel_regressor_ETfe = PanelOLS.from_formula("YIT ~ 1 + X1 + X2 + X3 + X4 + EntityEffects + TimeEffects", data=df)                          
print(panel_regressor_ETfe.fit())


'''Model Comparison'''
pool_regressor = pool_regressor.fit()  
panel_regressor_Tfe = panel_regressor_Tfe.fit()    
panel_regressor_Efe = panel_regressor_Efe.fit()
panel_regressor_ETfe = panel_regressor_ETfe.fit()

from linearmodels.panel import compare
print(compare({'Pooled':pool_regressor,
               'Time-fixed':panel_regressor_Tfe,
               'Entity-fixed':panel_regressor_Efe,
               'Both_T&E_Effects':panel_regressor_ETfe}))


######################################################################################################
######################################################################################################

import pandas as pd
import statsmodels.api as sm
from linearmodels.panel import PanelOLS

df = pd.read_csv('dairy.csv')
FARM = pd.Categorical(df.FARM)
YEAR = pd.Categorical(df.YEAR)
df = df.set_index(["FARM","YEAR"])
df['FARM'] = FARM
df['YEAR'] = YEAR


'''##### Q1: Pooled Regression #####'''
pool_regressor = PanelOLS.from_formula("YIT ~ 1 + COWS + LAND + LABOR + FEED", data=df)     
pool_regressor = pool_regressor.fit()                   
print(pool_regressor)

from linearmodels.panel import PooledOLS
import statsmodels.api as sm
exog_vars = ['COWS','LAND','LABOR','FEED']
exog = sm.add_constant(df[exog_vars])
mod = PooledOLS(df.YIT, exog)
pooled_regressor = mod.fit()
print(pooled_regressor)


'''##### Q2: Panel Regression fixed over time #####'''
from linearmodels.panel import PanelOLS
panel_regressor_Tfe = PanelOLS.from_formula("YIT ~ 1 + COWS + LAND + LABOR + FEED + TimeEffects", data=df)
panel_regressor_Tfe = panel_regressor_Tfe.fit()                      
print(panel_regressor_Tfe)


from linearmodels.panel import PanelOLS
import statsmodels.api as sm
exog = sm.add_constant(df[['COWS','LAND','LABOR','FEED']])
panel_regressor_Tfe = PanelOLS(df['YIT'], exog, entity_effects=False, time_effects=True)
print(panel_regressor_Tfe.fit())


from linearmodels.panel import PanelOLS
panel_regressor_Tfe = PanelOLS.from_formula("YIT ~ 1 + COWS + LAND + LABOR + FEED + YEAR", data=df)                          
print(panel_regressor_Tfe.fit())


from linearmodels.panel import PooledOLS
import statsmodels.api as sm
exog_vars = ['COWS','LAND','LABOR','FEED','YEAR']
exog = sm.add_constant(df[exog_vars])
mod = PooledOLS(df.YIT, exog)
pooled_Tfe = mod.fit()
print(pooled_Tfe)



'''##### Q3: Panel Regression fixed over farms #####'''
from linearmodels.panel import PanelOLS
panel_regressor_Efe = PanelOLS.from_formula("YIT ~ 1 + COWS + LAND + LABOR + FEED + EntityEffects", data=df)                          
panel_regressor_Efe = panel_regressor_Efe.fit()
print(panel_regressor_Efe)



'''##### Q4: Panel Regression with both time and farms effect #####'''
from linearmodels.panel import PanelOLS
panel_regressor_ETfe = PanelOLS.from_formula("YIT ~ 1 + COWS + LAND + LABOR + FEED + EntityEffects + TimeEffects", data=df)                          
panel_regressor_ETfe = panel_regressor_ETfe.fit()
print(panel_regressor_ETfe)

from linearmodels.panel import PanelOLS
panel_regressor_ETfe = PanelOLS.from_formula("YIT ~ 1 + COWS + LAND + LABOR + FEED + FARM + YEAR", data=df)                          
panel_regressor_ETfe = panel_regressor_ETfe.fit()
print(panel_regressor_ETfe)

'''Model Comparison'''
from linearmodels.panel import compare
print(compare({'Pooled':pool_regressor,
               'Time-fixed':panel_regressor_Tfe,
               'Entity-fixed':panel_regressor_Efe,
               'Both_T&E_Effects':panel_regressor_ETfe}))

######################################################################################################

