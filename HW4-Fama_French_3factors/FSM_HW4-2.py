# -*- coding: utf-8 -*-
"""
Created on Mon Oct  26 19:12:30 2020

@author: AllenPC
"""
import pandas as pd
import statsmodels.api as sm 

df = pd.read_csv('F-F_Research_Data_Factors_monthly.csv', index_col='yr_mth')
RF = df.loc['192701':'201712','RF']
three_factor = df.loc['192701':'201712','Mkt-RF':'HML']

X1 = sm.add_constant(three_factor) 
reg = sm.OLS(RF, X1).fit()
y_fitted = reg.fittedvalues
reg.summary()
