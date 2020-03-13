#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""  Project Summary 

Applied machine learning mathods to predict weekly ETF returns, regard these prediction as subjective views.
Incorporated views into Black-Litterman framework and constructed an optimal portfolio.
Back test the model to see whether it outperforms the result from mean variance optimization
"""

"""   Code Structure

My_main function is the main function used to test single machine leanring method to produce views. 
Including downloading data, process data, obtain features, make rolling prediction and backtest

Get_feature.py include feature engineering and feature selection

view.py is the script to produce subjective views by machine learning methods, which includes two type of prediction.
One is to predict the direction of return, Another one is to predict the rate of change.

Backtest.py is the code to evaluate the result of optimal portfolio, give various measures including Sharpe Ratio, 
max Drawdown.
"""

# ----------------------------------- Should change before run --------------------------------------------------

# Change the Pathname into path of data folder that I provide
Pathname = '/Users/jax/Downloads/searching jobs/company/MassMutual/code/data/'

''' important '''
# Need to pip install ta-lib library for feature engineering part
# If you have problems about download this library, please check this website:
#    https://mrjbq7.github.io/ta-lib/install.html 

# ---------------------------------------- Get Data & Parameters ------------------------------------------------

import pandas as pd
import numpy as np
import datetime

# Own code function
from Backtest import Backtest
from get_feature import feature_intro
from view import get_view

# Ticker name and feature_ticker name
tickers = ['EMB', 'GLD', 'TLT', 'IYR', 'IGIB', 'IJH']
weight4060 = np.matrix([0.15,0.15,0.2,0.15,0.2,0.15]).T
features = ['SPY', '^VIX', '^TNX', '^IRX']

# Get price data from CSV
databook = []
for i in tickers+features:
    databook += [pd.read_csv(Pathname + i+ '.csv',encoding='utf-8')]
#close = pdr.get_data_yahoo(tickers, start=st, end=end)["Adj Close"] # Original way: download data from Yahoo API

# Transfrom the format of date
date_list =list( databook[0]['Date'])
date = []
for i in range(len(date_list)):
    date += [datetime.datetime.strptime(date_list[i], "%Y-%m-%d")]

# Obtain close price 
close = []
for i in range(len(tickers)):
    close += [ databook[i]["Adj Close"]]
close = pd.DataFrame(np.matrix(close).T,columns = tickers ,index =date )
ret = close.pct_change().dropna()

## data visualization
#ret.describe()
#ret.plot()

# Get market weight data
market_weights = pd.read_csv(Pathname+'market_weights.csv', index_col=0)

# Define the function to get weight according to date 
def get_market_weight(date):
    start_str = str(date.year) + '/' + str(date.month) + '/' + str(date.day)
    weight = market_weights[market_weights.index == start_str]
    return np.matrix(weight).T


# ------------------------------------------ Get features ------------------------------------------

# Merge price information togther into a DataFrame for feature_into function
dataframe_columns = []
for i in range(len(tickers+features)):
    
    # Join tables together
    index_name = list(databook[i].columns)[1:]
    if i == 0 :
        price = np.matrix(databook[i][index_name])
    else:
        price = np.hstack((price , np.matrix(databook[i][index_name])))
    
    # Change column name by adding ticker name
    for j in range(len(index_name )):
        index_name[j] = (tickers+features)[i]+ index_name[j]   
    dataframe_columns += index_name

# Obtain price information of all tickers
price_info = pd.DataFrame(price,columns = dataframe_columns ,index =date )

# Apply feature_into function from get_feature.py to obtained selected features
feature_list_Y1, feature_list_Y2 = feature_intro(close, price_info)

# ------------------------------------------ parameter initialziation ------------------------------------------
class_type ='Logistic' # 'Random Forest' #'SVM'  #'Decision Tree'# 'KNN' #  # #  # 'Bayesian'##    
Ndata = 200   # How many training data each time to fix
window_rebalance = 1  # Rebalance every week
labda = 0.94
tao = 1 / (Ndata / 52)  # Following result of Meucci(2010)
# 7 year market risk aversion, can try points like 0.2, 0.5,1,2
risk_aversion = 0.5344


# ------------------------------------------  Running Strategy ------------------------------------------

# Initialization
total_ret = np.zeros(len(ret))
capital_ret = np.zeros(len(ret))
weight = pd.DataFrame(index = ret.index,columns = tickers)
final_result = np.zeros([3,6])
direct_ret = np.zeros(len(ret))

t = 0

# Take iteration from time series data, make rolling prediction and adjust portfolio weight every week
for i in range(Ndata+50, len(ret)-90, window_rebalance):

    # Use EWMA covariance matrix to estimate current cov
    cov_matrix = ret.iloc[i-Ndata:i, :].cov()
    last_day_return = np.matrix(ret.iloc[i-1,:] - ret.iloc[i-1,:].mean()).T
    
    # Adjust covariance matrix by adding more weights on current change
    current_cov = np.matrix(labda * cov_matrix + (1 - labda) * last_day_return * last_day_return.T)

    # Obtain market capital weight
    capital_weight = get_market_weight(ret.index[i-1])
    # Equivebriant return according to mean variance framework
    pi = np.matrix(risk_aversion * current_cov * capital_weight)

    # Apply machine learning method to produce subjective view on ETFs, get_view is from get_feature.oy
    view,result = get_view(feature_list_Y1, feature_list_Y2,class_type,ret.index[i],Ndata)
    final_result += result 
    t += 1
    
    # ---------------- Implementation of Black-litterman model 
    individual_var = np.diagonal(current_cov)
    
    # Subjective view on next week's return
    Q = pi + np.matrix(view * np.array(individual_var)).T
    # Subjectiv view on estimatimation variance
    omega = np.matrix(np.diag(individual_var))
    
    # Apply Bayesian approach to update the final estimation
    posterior_mean = pi + tao * current_cov * (current_cov * tao + omega).I * (Q - pi)
    # Exact formula should be pi + tao* np.dot(current_cov.I,Q - pi)
    posterior_sigma = current_cov + ((tao * current_cov).I + omega.I).I
    
    # Obtained optimal weight from Black-littermna model
    optimal_weight = posterior_sigma.I * posterior_mean / risk_aversion
    optimal_weight = optimal_weight / abs(optimal_weight).sum()   # Normalize
    
    weight.iloc[i,:] = optimal_weight.getA()[:,0].T
    
    # ------------ Calculate the performance of optimal portfolio and other contrast portfolios
    total_ret[i] = np.dot(ret.iloc[i, :], optimal_weight).getA()[0][0]
    
    # If view = 0, compute the performance of benchmarket portfolio
    capital_ret[i] = np.dot(ret.iloc[i, :], capital_weight).getA()[0][0]
    # Direct ret from machine learning
    direct_ret[i] = sum(ret.iloc[i, :] * view*0.1)

final_ret = pd.Series(total_ret, index=ret.index).dropna()
#plt.plot(net_value)

# --------------------------------- Backtest and compared with benchmark ---------------------------------------
# Used backtest.py to evaluate the portfolio and prediction performance

spy_ret = price_info['SPYClose'].pct_change().dropna()
all_ret = pd.DataFrame({'BL':final_ret,'Market':capital_ret,'SPY':spy_ret,'Direct_view':direct_ret})

# Get rid of all rows that ret = 0
df = all_ret[~all_ret['BL'].isin([0])]
net_value = np.cumprod(1+df)
net_value.iloc[:,0:3].plot()

# Evaluate the performance of portfolios
backtest = Backtest(df,'SPY',0.05,52)
result =backtest.summary()
print(result)

# Evaluate the accuracy of prediction
evaluation = pd.DataFrame(final_result/t,index=['sign_predict','range_predict','total_predict'],\
                          columns = tickers)
print('\n----------------------  prediction accuracy table -------------------')
print(evaluation)