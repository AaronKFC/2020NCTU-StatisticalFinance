# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 16:28:53 2020

@author: AllenPC
"""

import numpy as np
import pandas as pd
from pandas_datareader import data as wb
import matplotlib.pyplot as plt
from scipy import stats

from statsmodels.stats import weightstats as stests
from scipy.stats import ttest_1samp

df = pd.read_csv('appl_pg_wmt.csv', index_col='datadate')
df = df[['tic','prccd']]
df_aapl = df.loc[df.loc[:,'tic']=='AAPL']
df_aapl = df_aapl['prccd']
df_pg = df.loc[df.loc[:,'tic']=='PG']
df_pg = df_pg['prccd']
df_wmt = df.loc[df.loc[:,'tic']=='WMT']
df_wmt = df_wmt['prccd']

####################################################################
'''Problem1-1: Calculate mean daily return'''
ret_aapl = (df_aapl / df_aapl.shift(1)) - 1
ret_aapl = ret_aapl.dropna().rename('AAPL')
ret_pg = (df_pg / df_pg.shift(1)) - 1
ret_pg = ret_pg.dropna().rename('PG')
ret_wmt = (df_wmt / df_wmt.shift(1)) - 1
ret_wmt = ret_wmt.dropna().rename('WMT')

ret_aapl.head()
ret_aapl.describe()
ret_pg.head()
ret_pg.describe()
ret_wmt.head()
ret_wmt.describe()

########################################################################
'''Problem1-2: One sample Z-test'''
def One_sp_ztest(ret):
    ztest, pval = stests.ztest(ret, x2=None, value=0, 
                               alternative='two-sided')
    if pval<0.05:
        print("reject null hypothesis")
    else:
        print("accept null hypothesis")
    return pval

pval_z_aapl = One_sp_ztest(ret_aapl)
pval_z_pg = One_sp_ztest(ret_pg)
pval_z_wmt = One_sp_ztest(ret_wmt)

########################################################################
'''Problem1-3: One sample t-test'''
def One_sp_ttest(ret):
    tset, pval = ttest_1samp(ret, 0)
    print('p-values', pval)
    if pval < 0.05:    # alpha value is 0.05 or 5%
       print(" reject null hypothesis")
    else:
       print("accept null hypothesis")
    return pval

pval_t_aapl = One_sp_ttest(ret_aapl)
pval_t_pg = One_sp_ttest(ret_pg)
pval_t_wmt = One_sp_ttest(ret_wmt)

########################################################################
'''Problem1-4: Confidence Interval'''
def conf_interval_t(data,alpha,mean,sem): #std是sem的根號n倍，即std=sem*np.sqrt(n)
    interval_t = stats.t.interval(alpha, df=(len(data)-1),loc=mean, scale=sem)     #95%置信水平的区间
    return np.round(interval_t,8)

CIt95_aapl = conf_interval_t(ret_aapl, 0.95, ret_aapl.mean(), stats.sem(ret_aapl))
print(f"95% Confidence Interval (AAPL): {CIt95_aapl}")
CIt95_pg = conf_interval_t(ret_pg, 0.95, ret_pg.mean(), stats.sem(ret_pg))
print(f"95% Confidence Interval (PG): {CIt95_pg}")
CIt95_wmt = conf_interval_t(ret_wmt, 0.95, ret_wmt.mean(), stats.sem(ret_wmt))
print(f"95% Confidence Interval (WMT): {CIt95_wmt}")

# def conf_interval_n(alpha,mean,std):
#     interval_n = stats.norm.interval(alpha,mean,std)  #95%置信水平的区间
#     return np.round(interval_n,8)

# CIn95_aapl = conf_interval_n(0.95, ret_aapl.mean(), ret_aapl.std()) #注意:我認為應該也是要用sem才對，但網路上大家都用std，怪。
# print(f"95% Confidence Interval (n): {CIn95_aapl}")
########################################################################
########################################################################
'''Problem2-1: pair sample Z-test'''
# 檢定兩組樣本平均數是否相等=檢定兩組樣本平均數相減後是否等於0
def pair_sp_ztest(pair_diff):
    ztest, pval = stests.ztest(pair_diff, x2=None, value=0, 
                                   alternative='two-sided')
    print(float(pval))
    if pval<0.05:
        print("reject null hypothesis")
    else:
        print("accept null hypothesis")
    return pval

pval_pair_z_aapl_pg = pair_sp_ztest(ret_aapl-ret_pg)
pval_pair_z_pg_wmt = pair_sp_ztest(ret_pg-ret_wmt)
pval_pair_z_wmt_aapl = pair_sp_ztest(ret_wmt-ret_aapl)
    
########################################################################
'''Problem2-2: Pair sample t-test'''
def pair_sp_ttest(ret1, ret2):
    ttest, pval_pair = stats.ttest_rel(ret1, ret2)
    print(pval_pair)
    if pval_pair<0.05:
        print("reject null hypothesis")
    else:
        print("accept null hypothesis")
    return pval_pair

pval_pair_t_aapl_pg = pair_sp_ttest(ret_aapl, ret_pg)
pval_pair_t_pg_wmt = pair_sp_ttest(ret_pg, ret_wmt)
pval_pair_t_wmt_aapl = pair_sp_ttest(ret_wmt, ret_aapl)

########################################################################
########################################################################
'''Problem3: covariance and correlation matrix'''
df_3firms = pd.concat([ret_aapl, ret_pg, ret_wmt], axis=1)
df_3firms.cov()
df_3firms.corr(method='pearson')

########################################################################
'''Problem4: test whether these correlations are significantly differ from 0?'''
corr_aapl_pg, pval_aapl_pg = stats.pearsonr(ret_aapl, ret_pg)
corr_pg_wmt, pval_pg_wmt = stats.pearsonr(ret_wmt, ret_pg)
corr_wmt_aapl, pval_wmt_aapl = stats.pearsonr(ret_aapl, ret_wmt)
# correlation, p-value = scipy.stats.spearmanr(x, y)
# correlation, p-value = scipy.stats.kendalltau(x, y)



