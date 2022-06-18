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
# import yfinance as yf
# from pandas_datareader import data as wb
import pandas_datareader as pdr
import argparse

# parser = argparse.ArgumentParser()
# # parser.add_argument('--path', type=str, default='Tweets-DateSet-master', help='dir path')
# # parser.add_argument('--outpath', type=str,default='stock_data_test', help='output dir')
# parser.add_argument('--start', type=str, default='2000-01-02', help='start date') #其實應該從11-25開始抓，但是rise前五天會不一樣
# parser.add_argument('--end', type=str, default='2006-12-31', help='end date')

# args = parser.parse_args()
# df = pdr.DataReader(['AAPL','DIS','BA'], data_source='yahoo', start=args.start, end=args.end)
# # df = df["Adj Close"]
# df = df["Close"]

df = pd.read_csv('dow_com_2000_2006.csv', index_col='datadate')
comp = ['DIS','MCD','BA']
df = df[['tic','prccd']]
df_aapl = df.loc[df.loc[:,'tic']==comp[0]]
df_aapl = df_aapl['prccd']
df_pg = df.loc[df.loc[:,'tic']==comp[1]]
df_pg = df_pg['prccd']
df_wmt = df.loc[df.loc[:,'tic']==comp[2]]
df_wmt = df_wmt['prccd']

ret_aapl = (df_aapl / df_aapl.shift(1)) - 1
ret_aapl = ret_aapl.dropna().rename(comp[0])
ret_pg = (df_pg / df_pg.shift(1)) - 1
ret_pg = ret_pg.dropna().rename(comp[1])
ret_wmt = (df_wmt / df_wmt.shift(1)) - 1
ret_wmt = ret_wmt.dropna().rename(comp[2])

returns = pd.concat([ret_aapl, ret_pg, ret_wmt], axis=1)
# returns = (df / df.shift(1)) - 1
# returns = returns.dropna()
total_stock = len(returns.columns)


covariance_matrix = returns.cov() * 252
stocks_expected_return = returns.mean() * 252
stocks_weights = np.array([1/3, 1/3, 1/3])
portfolio_return = sum(stocks_weights * stocks_expected_return)
portfolio_risk = np.sqrt(reduce(np.dot, [stocks_weights, covariance_matrix, stocks_weights.T]))
print('個股平均收益(6年): ' +comp[0]+':' +str(round(stocks_expected_return[0],4)) +'  '
                           +comp[1]+':' +str(round(stocks_expected_return[1],4)) +'  '
                           +comp[2]+':' +str(round(stocks_expected_return[2],4)))
print('(權重均等)投資組合預期報酬率為: '+ str(round(portfolio_return,4)))
print('(權重均等)投資組合風險為: ' + str(round(portfolio_risk,4)))


##########################################################################
risk_list = []
return_list = []

stop = 0
while stop < 5000:#00:
    try:
        stop += 1
        weight = np.random.rand(total_stock)
        weight = weight / sum(weight)
        return_list.append(sum(stocks_expected_return * weight))
        risk_list.append(np.sqrt(reduce(np.dot, [weight, covariance_matrix, weight.T])))
    except:
        pass

fig = plt.figure(figsize = (10,6))
fig.suptitle('Stochastic simulation results', fontsize=18, fontweight='bold')
ax = fig.add_subplot()
ax.plot(risk_list, return_list, 'o')
ax.set_title('n=5000', fontsize=16)
fig.savefig('result.png',dpi=300)


def standard_deviation(weights):
    return np.sqrt(reduce(np.dot, [weights, covariance_matrix, weights.T]))


x0 = stocks_weights
bounds = tuple((0, 1) for x in range(total_stock))
constraints = [{'type': 'eq', 'fun': lambda x: sum(x) - 1}]
minimize_variance = solver.minimize(standard_deviation, x0=x0, constraints=constraints, bounds=bounds)
mvp_risk = minimize_variance.fun
mvp_return = sum(minimize_variance.x * stocks_expected_return)

print('風險最小化投資組合預期報酬率為:' + str(round(mvp_return,3)))
print('風險最小化投資組合風險為:' + str(round(mvp_risk,3)))

for i in range(total_stock):
    print(str(returns.columns[i])+' 佔投資組合權重 : ' + str(format(minimize_variance.x[i], '.4f')))

########################################################
'''效率前緣(Efficient Frontier)'''
x0 = stocks_weights #变量的初始猜测值
bounds = tuple((0, 1) for x in range(total_stock))

efficient_fronter_return_range = np.arange(0.06, 0.16, .003)
efficient_fronter_risk_list = []

for i in efficient_fronter_return_range:
    constraints = [{'type': 'eq', 'fun': lambda x: sum(x) - 1},
                   {'type': 'eq', 'fun': lambda x: sum(x * stocks_expected_return) - i}]
    efficient_fronter = solver.minimize(standard_deviation, x0=x0, 
                                        constraints=constraints, bounds=bounds)
    efficient_fronter_risk_list.append(efficient_fronter.fun)


risk_free = 0.02

fig = plt.figure(figsize = (12,6))
fig.subplots_adjust(top=0.85)
ax = fig.add_subplot()

fig.subplots_adjust(top=0.85)
ax0 = ax.scatter(risk_list, return_list,
                c=(np.array(return_list)-risk_free)/np.array(risk_list),
                marker = 'o')
ax.plot(efficient_fronter_risk_list, efficient_fronter_return_range, linewidth=1, color='#251f6b', marker='o',
         markerfacecolor='#251f6b', markersize=5)
ax.plot(mvp_risk, mvp_return,'*',color='r', markerfacecolor='#ed1313',  markersize=25)


ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_title('Efficient Frontier (' +comp[0]+' '+comp[1]+' '+comp[2] +')', fontsize=22, fontweight='bold')
ax.set_xlabel('Risk')
ax.set_ylabel('Return')
fig.colorbar(ax0, ax=ax, label = 'Sharpe Ratio')
plt.savefig('Efficient_Frontier.png',dpi=300)

########################################################################






