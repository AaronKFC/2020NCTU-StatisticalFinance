# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 21:52:21 2020

@author: AllenPC
"""

import pandas as pd
df = pd.read_stata('WAGE2.DTA')

from linearmodels import IV2SLS#, IVLIML, IVGMM, IVGMMCUE
df['const'] = 1
IVR = IV2SLS(dependent=df.wage, 
             exog=df.const, 
             endog=df.educ, 
             instruments=df.sibs).fit(cov_type='unadjusted')

print(IVR.summary)

###############################################################################
import statsmodels.api as sm 
X1 = sm.add_constant(df.educ) 
LR = sm.OLS(df.wage, X1).fit()
print(LR.summary())

Error_term = df.wage - (LR.params.const + LR.params.educ*df.educ)

from scipy import stats
corr, pval = stats.pearsonr(df.educ, Error_term**2)
print(f'correlation={corr:.4f}, p-value={pval:.4f}')

