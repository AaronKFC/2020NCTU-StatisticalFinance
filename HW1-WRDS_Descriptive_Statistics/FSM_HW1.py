# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 21:58:24 2020

@author: AllenPC
"""

import numpy as np
import pandas as pd
from pandas_datareader import data as wb
import matplotlib.pyplot as plt

mydata = pd.read_csv('DJI_close.csv')#, index_col='date')
mydata.info()
mydata = pd.Series(mydata['dji'].values, index=mydata['date'])

returns = (mydata / mydata.shift(1)) - 1

def return_hist(returns, mean, var, std, skew, kurt):
#    plt.hist(acc, bins=40, normed=0, facecolor="blue", edgecolor="black", alpha=0.7)
    plt.hist(returns, bins=80, density=True, facecolor='r',alpha=0.9)
    info = r'mean=%.4f, var=%.4f, std=%.4f' %(mean, var, std) 
    info2 = r'skew=%.4f, kurt=%.4f' %(skew, kurt) 
    plt.text(0.05, 40, info, ha='center', bbox=dict(facecolor='blue', alpha=0.25))
    plt.text(0.04, 35, info2, ha='center', bbox=dict(facecolor='blue', alpha=0.25))
    plt.xlabel('return')
    plt.ylabel('counts')
    plt.title("Return Distribution")
    plt.grid(True)
    plt.show()
#    plt.savefig(filename + '.png')

ret_mean = returns.mean()
ret_var = returns.var()
ret_std = returns.std()
ret_skew = returns.skew()
ret_kurt = returns.kurt()

return_hist(returns, ret_mean, ret_var, ret_std, ret_skew, ret_kurt)
#############################

def ret_pos_hist(returns, mean, var, std, skew, kurt):
    plt.hist(returns, bins=80, density=True, facecolor='b',alpha=0.9)
    info = r'mean=%.4f, var=%.4f, std=%.4f' %(mean, var, std) 
    info2 = r'skew=%.4f, kurt=%.4f' %(skew, kurt) 
    plt.text(0.03, 65, info, ha='center', bbox=dict(facecolor='green', alpha=0.25))
    plt.text(0.025, 56, info2, ha='center', bbox=dict(facecolor='green', alpha=0.25))
    plt.xlabel('return')
    plt.ylabel('counts')
    plt.title("Return Distribution")
    plt.grid(True)
    plt.show()
#    plt.savefig(filename + '.png')

ret_pos = returns.copy()
col = ret_pos[ret_pos <= 0].index
ret_pos.drop(col, inplace=True)
# df_copy.loc[df_copy[(0. <= df_copy['APC_1ST_YEARDIF']) & (df_copy['APC_1ST_YEARDIF'] <= per_25)].index, ['APC_1ST_YEARDIF']] = 4
# df_cp.drop(df_cp[df_cp['CHANNEL_B_POL_CNT'] > 4].index, inplace=True)

ret_pos_mean = ret_pos.mean()
ret_pos_var = ret_pos.var()
ret_pos_std = ret_pos.std()
ret_pos_skew = ret_pos.skew()
ret_pos_kurt = ret_pos.kurt()

ret_pos_hist(ret_pos, ret_pos_mean, ret_pos_var, ret_pos_std, ret_pos_skew, ret_pos_kurt)

