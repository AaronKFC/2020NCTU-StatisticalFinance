# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 19:12:30 2020

@author: AllenPC
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as solver
from functools import reduce
import pandas_datareader as pdr
import argparse
import statsmodels.api as sm 

df = pd.read_csv('dow_com_2000_2006.csv', index_col='datadate')
comp = ['DIS','MCD','BA']
df = df[['tic','prccd']]
df_1 = df.loc[df.loc[:,'tic']==comp[0]]
df_1 = df_1['prccd']
df_2 = df.loc[df.loc[:,'tic']==comp[1]]
df_2 = df_2['prccd']
df_3 = df.loc[df.loc[:,'tic']==comp[2]]
df_3 = df_3['prccd']

ret_1 = (df_1 / df_1.shift(1)) - 1
ret_1 = ret_1.dropna().rename(comp[0])
ret_2 = (df_2 / df_2.shift(1)) - 1
ret_2 = ret_2.dropna().rename(comp[1])
ret_3 = (df_3 / df_3.shift(1)) - 1
ret_3 = ret_3.dropna().rename(comp[2])

returns = pd.concat([ret_1, ret_2, ret_3], axis=1)
total_stock = len(returns.columns)

covariance_matrix = returns.cov() * 252
stocks_expected_return = returns.mean() * 252
stocks_weights = np.array([1/3, 1/3, 1/3])#, .1,.1,.1,.1,.1,.1,.1])
portfolio_return = sum(stocks_weights * stocks_expected_return)
portfolio_risk = np.sqrt(reduce(np.dot, [stocks_weights, covariance_matrix, stocks_weights.T]))
print('個股平均收益(6年): ' +comp[0]+':' +str(round(stocks_expected_return[0],4)) +'  '
                           +comp[1]+':' +str(round(stocks_expected_return[1],4)) +'  '
                           +comp[2]+':' +str(round(stocks_expected_return[2],4)))
print('(平均配置)投資組合預期報酬率為: '+ str(round(portfolio_return,4)))
print('(平均配置)投資組合風險為: ' + str(round(portfolio_risk,4)))


##########################################################################

def standard_deviation(weights):
    return np.sqrt(reduce(np.dot, [weights, covariance_matrix, weights.T]))

########################################################
'''效率前緣(Efficient Frontier)'''
x0 = stocks_weights
bounds = tuple((0, 1) for x in range(total_stock))

efficient_fronter_return_range = np.arange(0.02, 0.18, .001)
efficient_fronter_risk_list = []
portfolio_weights = []

for i in efficient_fronter_return_range:
    constraints1 = [{'type': 'eq', 'fun': lambda x: sum(x) - 1},
                   {'type': 'eq', 'fun': lambda x: sum(x * stocks_expected_return) - i}]
    efficient_fronter = solver.minimize(standard_deviation, x0=x0, constraints=constraints1, bounds=bounds)
    efficient_fronter_risk_list.append(efficient_fronter.fun)
    portfolio_weights.append(efficient_fronter.x)

'''資本市場線(Capital Market Line)'''
x0 = stocks_weights
bounds = tuple((0, 1) for x in range(total_stock))

# efficient_fronter_return_range = np.arange(0.02, 0.15, .003)
CML_risk_list = []

for i in efficient_fronter_return_range:
    constraints2 = [{'type': 'eq', 'fun': lambda x: 0.02 + sum(x * (stocks_expected_return-0.02)) - i}]
    efficient_fronter = solver.minimize(standard_deviation, x0=x0, constraints=constraints2, bounds=bounds)
    CML_risk_list.append(efficient_fronter.fun)
    # portfolio_weights.append(efficient_fronter.x)
    
'''切點(Tangency Portfolio)'''
TP_idx = np.argmin(abs(np.array(CML_risk_list) - np.array(efficient_fronter_risk_list)))
[TP_x, TP_y] = [CML_risk_list[TP_idx], efficient_fronter_return_range[TP_idx]]
### 切點投資組合如下
#   切點投資組合預期報酬率為: TP_y
#   切點投資組合風險為: TP_x
print('切點投資組合預期報酬率為:' + str(round(TP_y,3)))
print('切點投資組合風險為:' + str(round(TP_x,3)))
Tangency_portfolio_weights = portfolio_weights[TP_idx]
for i in range(total_stock):
    print(str(returns.columns[i])+' 佔投資組合權重 : ' + str(round(Tangency_portfolio_weights[i],4)))


###################
risk_free = 0.02
Rf = 0.02/252

portfolio_Rm = np.sum(Tangency_portfolio_weights * returns,axis=1)
Rm_Rf = portfolio_Rm - Rf
ret1_Rf = ret_1 - Rf

X1 = sm.add_constant(Rm_Rf) #使用 sm.add_constant() 在 array 上加入一列常项1。
reg = sm.OLS(ret1_Rf, X1).fit()
y_fitted = reg.fittedvalues
reg.summary()

fig = plt.figure(figsize = (10,6))
fig_fn = 'CAPM Linear Regression (' +comp[0]+')'
ax = fig.add_subplot()
ax.plot(Rm_Rf, ret1_Rf, 'o')
ax.set_title(fig_fn, fontsize=18, fontweight='bold')
# ax.plot(Rm_Rf, beta * Rm_Rf + alpha, '-', color = 'r')
ax.plot(Rm_Rf, y_fitted, '-', color = 'r')
fig.savefig(fig_fn+'.png',dpi=300)

###################
portfolio_Rm = np.sum(Tangency_portfolio_weights * returns,axis=1)
Rm_Rf = portfolio_Rm - Rf
ret2_Rf = ret_2 - Rf

X1 = sm.add_constant(Rm_Rf) #使用 sm.add_constant() 在 array 上加入一列常项1。
reg = sm.OLS(ret2_Rf, X1).fit()
y_fitted = reg.fittedvalues
reg.summary()

fig = plt.figure(figsize = (10,6))
fig_fn = 'CAPM Linear Regression (' +comp[1]+')'
ax = fig.add_subplot()
ax.plot(Rm_Rf, ret2_Rf, 'o')
ax.set_title(fig_fn, fontsize=18, fontweight='bold')
# ax.plot(Rm_Rf, beta * Rm_Rf + alpha, '-', color = 'r')
ax.plot(Rm_Rf, y_fitted, '-', color = 'r')
fig.savefig(fig_fn+'.png',dpi=300)

###################
portfolio_Rm = np.sum(Tangency_portfolio_weights * returns,axis=1)
Rm_Rf = portfolio_Rm - Rf
ret3_Rf = ret_3 - Rf

X1 = sm.add_constant(Rm_Rf) #使用 sm.add_constant() 在 array 上加入一列常项1。
reg = sm.OLS(ret3_Rf, X1).fit()
y_fitted = reg.fittedvalues
reg.summary()

fig = plt.figure(figsize = (10,6))
fig_fn = 'CAPM Linear Regression (' +comp[2]+')'
ax = fig.add_subplot()
ax.plot(Rm_Rf, ret3_Rf, 'o')
ax.set_title(fig_fn, fontsize=18, fontweight='bold')
# ax.plot(Rm_Rf, beta * Rm_Rf + alpha, '-', color = 'r')
ax.plot(Rm_Rf, y_fitted, '-', color = 'r')
fig.savefig(fig_fn+'.png',dpi=300)
########################################################################
# beta, alpha = np.polyfit(ret_1, Rm_Rf, 1)
# print(f'Beta for {comp[0]} stock is = {beta:.4f} and alpha is = {alpha:.4f}')#.format(comp[0], beta, alpha))

