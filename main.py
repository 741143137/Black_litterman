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
Pathname = '/Users/jax/Downloads/study/796/final project/codes/'

st = datetime.datetime(2008,1,1)
end = datetime.datetime(2015,12,31)
#out sample
#st = datetime.datetime(2014,10,31)
#end = datetime.datetime(2018,10,31)

# Get price data
close = pdr.get_data_yahoo(tickers,start = st,end = end)["Adj Close"]
ret = close.pct_change().dropna()
# Get market weight data
market_weights = pd.read_csv(Pathname + 'market_weights.csv',index_col=0)



# =========================== Data processing =================================================
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

def get_market_weight(date):
    start_str = str(date.year)+'/'+ str(date.month)+'/'+str(date.day)
    weight = market_weights[market_weights.index == start_str]
    return np.matrix(weight).T

# ============================= Train Data using Classification==============================







# ===================================================
''' parameter initialziation '''
N_length = 250
window_rebalance = 5             # Rebalance every week
labda = 0.94
tao = 1/ (N_length / 250)        # Following result of Meucci(2010)           
# 7 year market risk aversion, can try points like 0.2, 0.5,1,2
risk_aversion = 0.5344           

total_ret = np.zeros(len(ret))

test_pi = []

for i in range(N_length,len(ret),window_rebalance):
    
    # There are two different ways to estimate current cov
    # 1 is use sample cov, 2 is use EWMA covariance matrix
    cov_matrix = ret.iloc[i-N_length:i,:].cov()
    
    last_day_return = np.matrix(ret.iloc[i-1,:] - ret.iloc[i-1,:].mean()).T
    current_cov = np.matrix(labda*cov_matrix + (1-labda)*last_day_return*last_day_return.T)
    
    # Obtain market capital weight
    capital_weight = get_market_weight(ret.index[i-1])
    # Equivebriant return 
    pi = np.matrix(risk_aversion*current_cov*capital_weight)
    
    ''' test test_pi.append(pi)'''
    
    # Get subjective view by machine learning classification
    view = 0#Get_view(feature,y1,y2)
    
    individual_var = np.diagonal(current_cov)
    Q = pi + view * np.matrix(individual_var**0.5).T
    
    # Note, since we 
    omega = np.matrix(np.diag(individual_var))
    
    posterior_mean = pi + tao*current_cov * (current_cov*tao+omega).I * (Q - pi)
    # Exact formula should be pi + tao* np.dot(current_cov.I,Q - pi)
    posterior_sigma = current_cov + ( (tao*current_cov).I + omega.I ).I
    
    optimal_weight = posterior_sigma.I * posterior_mean / risk_aversion
    
    #calculate return
    
    if i+window_rebalance < len(ret):
        total_ret[i:i+window_rebalance] = np.dot(ret.iloc[i:i+window_rebalance,:],optimal_weight).getA()[:,0]
    else:
        total_ret[i:] = np.dot(ret.iloc[i:i+window_rebalance,:],optimal_weight).getA()[:,0]

net_value = np.cumprod(1+total_ret)

    