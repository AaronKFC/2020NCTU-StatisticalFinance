# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 11:01:08 2020

@author: AllenPC
"""

#load all necessary libraries
import pandas as pd 
import numpy as np 
# import scipy as scp
import sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn import metrics 
from sklearn.metrics import confusion_matrix, accuracy_score

################################################################################################
'''########## Question 1 ##########'''
df = pd.read_table('data.txt', header=None, delim_whitespace=True, 
                   names=['Date', 'Temp', 'Conct', 'Damage', 'Count'])

# df.describe()
# print(df['Temp'].value_counts())
# print(df['Conct'].value_counts())
# print(df['Count'].value_counts())

df1 = df.copy()
df1.loc[df1[(df1['Damage']>0)].index, ['Damage']] = 1

X1 = df1.drop(['Damage'], axis=1) 
y1 = df1['Damage']
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size = 0.20, random_state = 5)
# print(X1_train.shape)
# print(X1_test.shape)
# print(y1_train.shape)
# print(y1_test.shape)

# '''Feature Scaling'''
# from sklearn.preprocessing import StandardScaler
# sc_X = StandardScaler()
# X1_train = sc_X.fit_transform(X1_train)
# X1_test = sc_X.transform(X1_test)

'''Method1: Use sklearn'''
LR1_sk = LogisticRegression(random_state=0, penalty='none', solver='newton-cg').fit(X1_train, y1_train)
pred_LR1_sk = LR1_sk.predict(X1_test)
LR1_sk.coef_

'''Method2: Use statsmodels'''
import statsmodels.api as sm
LR1_sm = sm.Logit(y1_train, X1_train).fit() 
LR1_sm.summary()
yhat = LR1_sm.predict(X1_test)
pred_LR1_sm = list(map(round, yhat)) 
print(pred_LR1_sm)

'''Model Evaluation'''
# Method1
confusion_matrix(y1_test, pred_LR1_sk)
accuracy_score(y1_test, pred_LR1_sk)

# Method2
confusion_matrix(y1_test, pred_LR1_sm)
accuracy_score(y1_test, pred_LR1_sm)


###################################################################################################################
'''########## Question 2 ##########'''
df2 = df.copy()
X2 = df2.drop(['Damage'], axis=1) 
y2 = df2['Damage']
X2_train, X2_test, y2_train, y2_test = sklearn.model_selection.train_test_split(X2, y2, test_size = 0.20, random_state = 5)
# print(X2_train.shape)
# print(X2_test.shape)
# print(y2_train.shape)
# print(y2_test.shape)

'''Method1: Use sklearn'''
LR2_sk = LogisticRegression(random_state=0, multi_class='multinomial', penalty='none', solver='newton-cg').fit(X2_train, y2_train)
pred_LR2_sk = LR2_sk.predict(X2_test)

#print the tunable parameters (They were not tuned in this example, everything kept as default)
params = LR2_sk.get_params()
print(params)

#Print model parameters
print('Intercept: \n', LR2_sk.intercept_)
print('Coefficients: \n',LR2_sk.coef_)

#Calculate odds ratio estimates
np.exp(LR2_sk.coef_)


'''Method2: Use statsmodels'''
LR2_sm=sm.MNLogit(y2_train,sm.add_constant(X2_train))
result=LR2_sm.fit()
stats1=result.summary()
print(stats1)
stats2=result.summary2()
print(stats2)

from statsmodels.tools import add_constant
pred_LR2 = result.predict(add_constant(X2_test[['Date', 'Temp', 'Conct', 'Count']]), transform=False)
# print(pred_LR2)
pred_LR2_sm = np.array(pred_LR2).argmax(1)
print(pred_LR2_sm)


'''Model Evaluation'''
# Method1
confusion_matrix(y2_test, pred_LR2_sk)
accuracy_score(y2_test, pred_LR2_sk)

# Method2
confusion_matrix(y2_test, pred_LR2_sm)
accuracy_score(y2_test, pred_LR2_sm)

###########################################################################################
'''########## Question 3 ##########'''
df3 = df.copy()
X3 = df3.drop(['Damage'], axis=1) 
y3 = df3['Damage']
X3_train, X3_test, y3_train, y3_test = sklearn.model_selection.train_test_split(X3, y3, test_size = 0.20, random_state = 5)

df3_train = pd.concat([X3_train, y3_train],axis=1)
df3_test = pd.concat([X3_test, y3_test],axis=1)

'''method 1-A'''
from statsmodels.miscmodels.ordinal_model import OrderedModel
LR3_sm = OrderedModel.from_formula("Damage ~ 0 + Date + Temp + Conct + Count", df3_train, distr='logit')
LR3_sm_ord = LR3_sm.fit(method='bfgs')
LR3_sm_ord.summary()
pred_LR3_sm_ord = LR3_sm_ord.model.predict(LR3_sm_ord.params, exog=X2_test[['Date', 'Temp', 'Conct', 'Count']])
# print(pred_LR2_sm_ord)
pred_LR3_sm_ord = pred_LR3_sm_ord.argmax(1)
print(pred_LR3_sm_ord)

'''method 1-B (same as 1-A)'''
mod_log = OrderedModel(df3_train['Damage'],
                       df3_train[['Date', 'Temp', 'Conct', 'Count']],
                       distr='logit')
res_log = mod_log.fit(method='bfgs', disp=False)
res_log.summary()
predicted = res_log.model.predict(res_log.params, exog=X2_test[['Date', 'Temp', 'Conct', 'Count']])
# print(predicted)
pred_choice = predicted.argmax(1)
print(pred_choice)


'''method 2'''
mod_prob = OrderedModel(df3_train['Damage'],
                        df3_train[['Date', 'Temp', 'Conct', 'Count']],
                        distr='probit')
res_prob = mod_prob.fit(method='bfgs')
res_prob.summary()
predicted_prob = res_prob.model.predict(res_prob.params, exog=X2_test[['Date', 'Temp', 'Conct', 'Count']])
# print(predicted_prob)
pred_choice_prob = predicted_prob.argmax(1)
print(pred_choice_prob)


'''Model Evaluation'''
# Method1
confusion_matrix(y2_test, pred_LR3_sm_ord)
accuracy_score(y2_test, pred_LR3_sm_ord)

# Method2
confusion_matrix(y2_test, pred_choice_prob)
accuracy_score(y2_test, pred_choice_prob)

#############################################################################################

'''Method 3-A (mord)'''
import mord
mul_lr = mord.OrdinalRidge(alpha=0.001, fit_intercept=True, normalize=False, 
                           copy_X=True, max_iter=None, tol=0.001, 
                           solver='auto').fit(X3_train, y3_train) 
pred_mord1 = mul_lr.predict(X3_test)
mul_lr.coef_ 
np.exp(mul_lr.coef_)

'''Method 3-B (mord)'''
from mord import LogisticAT
model_ordinal = LogisticAT(alpha=0).fit(X3_train, y3_train) 
pred_mord2 = model_ordinal.predict(X3_test)
model_ordinal.coef_


'''Model Evaluation'''
# Method1
confusion_matrix(y3_test, pred_mord1)
accuracy_score(y3_test, pred_mord1)

# Method2
confusion_matrix(y3_test, pred_mord2)
accuracy_score(y3_test, pred_mord2)

#############################################################################################
