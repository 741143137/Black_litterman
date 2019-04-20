#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 09:20:50 2019

@author: jax
"""


#-------------------------------------------- Get Data & Parameters ------------------------------------------------

"""
1.0 Initialization
"""
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
pd.core.common.is_list_like = pd.api.types.is_list_like
import fix_yahoo_finance as yf
from pandas_datareader import data as pdr      # For download data

yf.pdr_override()
tickers = ['EMB','GLD','TLT','IYR','IGIB','IJH']
BlackRock_adjust = [0.25,0.03,0.04,0.08,0.15,0.24,0.04,0.12,0.05]

st = datetime.datetime(2007,1,1)
end = datetime.datetime(2015,10,31)
#out sample
#st = datetime.datetime(2014,10,31)
#end = datetime.datetime(2018,10,31)

#close = pd.DataFrame(columns = tickers)
close = pdr.get_data_yahoo(tickers,start = st,end = end)["Adj Close"]
ret = close.pct_change().dropna()

window = 3
N = len(ret)
M = len(tickers)
y = np.zeros([N-3,M])
rolling_sd = ret.rolling(window=window).std()
z2 = abs((ret - ret.shift(-1))/rolling_sd)
y1 = np.zeros([N,M])
y1[ret<0] = -1
y1[ret>0] = 1

# compute how much stock return change
y2 = np.zeros([N,M])
y2[z2<=1] = 1
y2[z2>1] = 2

print('The data right now is ret, y1, y2')

# ============================= Train Data using Classification==============================







# ===================================================
''' parameter initialziation '''
N_length = 250
window_rebalance = 5             # Rebalance every week
labda = 0.94
tao = 1/ (N_length / 250)        # Following result of Meucci(2010)           
risk_aversion = 2

total_ret = np.zeros(len(ret)-N_length)

for i in range(N_length,len(ret),window_rebalance):
    
    # There are two different ways to estimate current cov
    # 1 is use sample cov, 2 is use EWMA covariance matrix
    cov_matrix = ret.iloc[i-N_length:i,:].cov()
    
    last_day_return = np.matrix(ret.iloc[i-1,:] - ret.iloc[i-1,:].mean()).T
    current_cov = labda*cov_matrix + (1-labda)*last_day_return*last_day_return.T
    
    # Equivebriant return 
    pi = risk_aversion*np.dot(current_cov,capital_weight(i))
    
    # Get subjective view by machine learning classification
    view = Get_view()
    
    Q = pi + view * diag(cov_matrix)
    
    posterior_mean = pi + tao* np.dot((current_cov*tao+omega).I,Q - pi)
    # Exact formula should be pi + tao* np.dot(current_cov.I,Q - pi)
    posterior_sigma = current_cov + ( (tao*current_cov).I + omega.I ).I
    
    optimal_weight = posterior_sigma.I * posterior_mean / risk_aversion
    
    #calculate return
    total_ret[i:i+window_rebalance] = optimal_weight * ret.iloc[i:i+window_rebalance,:]
    



    