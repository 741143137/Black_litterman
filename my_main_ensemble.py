#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 20:37:26 2019
"""

'''
My_main function is the main function used to test all machine leanring methods together
Including downloading data, process data, obtain features, make rolling prediction and backtest
''' 


# -------------------------------------------- Get Data & Parameters ------------------------------------------------

"""
1.0 Initialization
"""
import pandas as pd
import numpy as np
import datetime

# Own code function
from Backtest import Backtest
from feature_select import feature_intro
from view import get_view


#yf.pdr_override()
# Six ETF to test
tickers = ['EMB', 'GLD', 'TLT', 'IYR', 'IGIB', 'IJH']
features = ['SPY', '^VIX', '^TNX', '^IRX']
Pathname = '/Users/jax/Downloads/study/796/final project/codes/weekly data/'

# Get price data
databook = []
for i in tickers+features:
    databook += [pd.read_csv(Pathname + i+ '.csv',encoding='utf-8')]

#close = pdr.get_data_yahoo(tickers, start=st, end=end)["Adj Close"]

close = []
for i in range(len(tickers)):
    close += [ databook[i]["Adj Close"]]
close = np.matrix(close).T
date_list =list( databook[0]['Date'])
date = []
for i in range(len(date_list)):
    date += [datetime.datetime.strptime(date_list[i], "%Y-%m-%d")]


close = pd.DataFrame(close,columns = tickers ,index =date )
ret = close.pct_change().dropna()

# Get market weight data
market_weights = pd.read_csv(Pathname+'market_weights.csv', index_col=0)

# =========================== Data processing =================================================
print('The data right now is ret, y1, y2')


def get_market_weight(date):
    start_str = str(date.year) + '/' + str(date.month) + '/' + str(date.day)
    weight = market_weights[market_weights.index == start_str]
    return np.matrix(weight).T


# ============================ Get features ================================================


#close2 = pdr.get_data_yahoo(features, start=st, end=end)["Adj Close"]
#close2_weekly =close2.iloc[[ i*5-1 for i in range(1,dayamount+1)]]

dataframe_columns = []
for i in range(len(tickers+features)):

    index_name = list(databook[i].columns)[1:]
    if i == 0 :
        price = np.matrix(databook[i][index_name])
    else:
        price = np.hstack((price , np.matrix(databook[i][index_name])))
    for j in range(len(index_name )):
        index_name[j] = (tickers+features)[i]+ index_name[j]
    dataframe_columns += index_name
price = pd.DataFrame(price,columns = dataframe_columns ,index =date )

feature_list_Y1, feature_list_Y2 = feature_intro(close, price)

# ===================================================
#''' parameter initialziation '''
class_type = 'Logistic'#'Random Forest' #'Bayesian' #'SVM' # 'KNN'    # 'Bayesian'##    
Ndata = 200   # How many training data each time to fix
window_rebalance = 1  # Rebalance every week
labda = 0.94
tao = 1 / (Ndata / 52)  # Following result of Meucci(2010)
# 7 year market risk aversion, can try points like 0.2, 0.5,1,2
risk_aversion = 0.5344


###########################################################################################

capital_ret = np.zeros(len(ret))
weight = pd.DataFrame(index = ret.index,columns = tickers)
final_result = np.zeros([3,6])
t = 0

type_list = ['Random Forest','Bayesian','SVM','Logistic','KNN','Adaboosting']
#type_list = ['Random Forest','SVM','Logistic']
total_ret = pd.DataFrame(index = ret.index,columns = type_list)
total_view = 0

for i in range(Ndata+50, len(ret)-90, window_rebalance):
#for i in range(len(ret)-90,len(ret), window_rebalance):  # This one is for out-of-sample-test
    # There are two different ways to estimate current cov
    # 1 is use sample cov, 2 is use EWMA covariance matrix
    cov_matrix = ret.iloc[i-Ndata:i, :].cov()

    last_day_return = np.matrix(ret.iloc[i-1,:] - ret.iloc[i-1,:].mean()).T
    #print( last_day_return)
    current_cov = np.matrix(labda * cov_matrix + (1 - labda) * last_day_return * last_day_return.T)

    # Obtain market capital weight
    capital_weight = get_market_weight(ret.index[i-1])
    # Equivebriant return
    pi = np.matrix(risk_aversion * current_cov * capital_weight)

    # multi_class
    t +=1
    
    for class_type in type_list:
        view,result = get_view(feature_list_Y1,feature_list_Y2,class_type,ret.index[i],Ndata)
        
        # Record total view as model ensamble
        total_view += view
        final_result += result
    
        individual_var = np.diagonal(current_cov)
        ''' mapping function '''
        #Q = pi + np.matrix(view * np.array(individual_var ** 0.5) * 0.2).T
        Q = pi + np.matrix(view * np.array(individual_var)).T
        # Note, since we
        omega = np.matrix(np.diag(individual_var))

        posterior_mean = pi + tao * current_cov * (current_cov * tao + omega).I * (Q - pi)
        # Exact formula should be pi + tao* np.dot(current_cov.I,Q - pi)
        posterior_sigma = current_cov + ((tao * current_cov).I + omega.I).I

        optimal_weight = posterior_sigma.I * posterior_mean / risk_aversion
        optimal_weight = optimal_weight / abs(optimal_weight).sum()
    
        weight.iloc[i,:] = optimal_weight.getA()[:,0].T
    
        total_ret[class_type][i] = np.dot(ret.iloc[i, :], optimal_weight).getA()[0][0]
    
    # If view = 0
    capital_ret[i] = np.dot(ret.iloc[i, :], capital_weight).getA()[0][0]

total_ret.dropna(inplace = True)
net_value = np.cumprod(1+total_ret)
net_value.plot()


## Backtest and benchmark
spy_ret = price['SPYClose'].pct_change().dropna()
market_ret = pd.Series(capital_ret,index=ret.index,name='market')
df = pd.concat([total_ret,market_ret,spy_ret],axis=1,join = 'inner')

## Get rid of all rows that ret = 0
#df = all_ret[~all_ret['BL'].isin([0])]
net_value = np.cumprod(1+df)
net_value.iloc[:,0:4].plot()
#
backtest = Backtest(df,'SPYClose',0.05,52)
#print(backtest.summary())
#evaluation = pd.DataFrame(final_result/t,index=['sign_predict','range_predict','total_predict'],\
#                          columns = tickers)
#evaluation['mean'] = np.mean(evaluation,axis=1)
#print(evaluation)