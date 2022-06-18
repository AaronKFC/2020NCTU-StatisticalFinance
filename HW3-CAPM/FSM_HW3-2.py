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
total_stock = len(returns.columns)


covariance_matrix = returns.cov() * 252
stocks_expected_return = returns.mean() * 252
stocks_weights = np.array([1/3, 1/3, 1/3])#, .1,.1,.1,.1,.1,.1,.1])
portfolio_return = sum(stocks_weights * stocks_expected_return)
portfolio_risk = np.sqrt(reduce(np.dot, [stocks_weights, covariance_matrix, stocks_weights.T]))
print('個股平均收益(6年): ' +comp[0]+':' +str(round(stocks_expected_return[0],4)) +'  '
                           +comp[1]+':' +str(round(stocks_expected_return[1],4)) +'  '
                           +comp[2]+':' +str(round(stocks_expected_return[2],4)))
print('((權重均等)投資組合預期報酬率為: '+ str(round(portfolio_return,4)))
print('(權重均等)投資組合風險為: ' + str(round(portfolio_risk,4)))


##########################################################################
risk_list = []
return_list = []

stop = 0
while stop < 5000:
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

########################################################
'''效率前緣(Efficient Frontier)'''
x0 = stocks_weights #变量的初始猜测值
bounds = tuple((0, 1) for x in range(total_stock))

efficient_fronter_return_range = np.arange(0.02, 0.18, .001)
efficient_fronter_risk_list = []
portfolio_weights = []
for i in efficient_fronter_return_range:
    constraints1 = [{'type': 'eq', 'fun': lambda x: sum(x) - 1},
                   {'type': 'eq', 'fun': lambda x: sum(x * stocks_expected_return) - i}]
    efficient_fronter = solver.minimize(standard_deviation, x0=x0, 
                                        constraints=constraints1, bounds=bounds)
    efficient_fronter_risk_list.append(efficient_fronter.fun)
    portfolio_weights.append(efficient_fronter.x)

'''資本市場線(Capital Market Line)'''
x0 = stocks_weights
bounds = tuple((0, 1) for x in range(total_stock))
CML_risk_list = []
for i in efficient_fronter_return_range:
    constraints2 = [{'type': 'eq', 
                     'fun': lambda x: 0.02 + sum(x * (stocks_expected_return-0.02)) - i}]
    efficient_fronter = solver.minimize(standard_deviation, x0=x0, 
                                        constraints=constraints2, bounds=bounds)
    CML_risk_list.append(efficient_fronter.fun)
        

'''切點(Tangency Portfolio)'''
TP_idx = np.argmin(abs(np.array(CML_risk_list) - np.array(efficient_fronter_risk_list)))
[TP_x, TP_y] = [CML_risk_list[TP_idx], efficient_fronter_return_range[TP_idx]]
print('切點投資組合預期報酬率為:' + str(round(TP_y,3)))
print('切點投資組合風險為:' + str(round(TP_x,3)))
for i in range(total_stock):
    print(str(returns.columns[i])+' 佔投資組合權重 : ' + str(round(portfolio_weights[TP_idx][i],4)))


###################
risk_free = 0.02

fig = plt.figure(figsize = (12,6))
fig.subplots_adjust(top=0.85)
ax = fig.add_subplot()

fig.subplots_adjust(top=0.85)
ax0 = ax.scatter(risk_list, return_list,
                c=(np.array(return_list)-risk_free)/np.array(risk_list),
                marker = 'o')
# Efficient Frontier
ax.plot(efficient_fronter_risk_list, efficient_fronter_return_range, linewidth=1, color='#251f6b', marker='o',
         markerfacecolor='#251f6b', markersize=5)

# Capital Market Line
ax.plot(CML_risk_list, efficient_fronter_return_range, linewidth=1, color='m', marker='o',
         markerfacecolor='m', markersize=5)

# 
ax.plot(TP_x, TP_y,'*',color='r', markerfacecolor='#ed1313',  markersize=25)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_title('Tangency Portfolio (' +comp[0]+' '+comp[1]+' '+comp[2] +')', fontsize=22, fontweight='bold')
ax.set_xlabel('Risk')
ax.set_ylabel('Return')
fig.colorbar(ax0, ax=ax, label = 'Sharpe Ratio')
plt.savefig('Tangency Portfolio.png',dpi=300)

########################################################################
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.optimize import leastsq
 
#  # 待拟合的数据
# Y = np.array(efficient_fronter_return_range)
# X = np.array(efficient_fronter_risk_list)
 
# # 二次函数的标准形式
# def func(params, y):
#  a, b, c = params
#  return a * y * y + b * y + c
 
# # 误差函数，即拟合曲线所求的值与实际值的差
# def error(params, x, y):
#  return func(params, y) - x
 
 
# # 对参数求解
# def slovePara():
#  p0 = [1, 1, 1]
 
#  Para = leastsq(error, p0, args=(X, Y))
#  return Para
 
 
# # 输出最后的结果
# def solution():
#  Para = slovePara()
#  a, b, c = Para[0]
#  print("a=",a," b=",b," c=",c)
#  print("cost:" + str(Para[1]))
#  print("求解的曲线是:")
#  print("x="+str(round(a,2))+"y*y+"+str(round(b,2))+"y+"+str(c))
 
#  plt.figure(figsize=(8,6))
#  plt.scatter(X, Y, color="green", label="sample data", linewidth=2)
 
#  # 画拟合直线
#  y=np.linspace(0.02,0.15,30) ##在0-15直接画100个连续点
#  x=a*y*y+b*y+c ##函数式
#  plt.plot(x,y,color="red",label="solution line",linewidth=2)
#  plt.legend() #绘制图例
#  plt.show()
 
 
# solution()
