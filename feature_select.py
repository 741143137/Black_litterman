#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 21:16:13 2019

@author: jax
"""

'''
feature_select.py is used to compute and select features for each ETF
final return is a list contains six dataframe
'''

import pandas as pd
import copy
import numpy as np

import sklearn as sk
from sklearn.ensemble import RandomForestClassifier
import talib as tl

# One change: technical indicator become t-1

# Previous_returns is the function to get lag terms of  
# time series data, n is how long the window is
# j is the lag term
def previous_returns(df, n, j):
    #n weeks, j lags
    #returns = df.pct_change(periods = 5*n)
    returns = df.pct_change(periods = n)
    returns = returns.shift(j)

    return returns


# feature_into compute total 105 features for six ETF
# close is close price for six ETF
# price is price information like open,high,low for ETF and other features like SPY, VIX

def feature_intro(close, price):

    returns = close.pct_change()
    feature_list = []
    for name in close.columns:
        data = pd.DataFrame(close[name])
        data.columns = [name + '_close']
        data[name + '_volume'] = price[ name+'Volume']
        data[name + '_returns'] = returns[name]
        data[name + '_Open'] = price[name+'Open']
        data[name + '_High'] = price[name+'High']
        data[name + '_Low'] = price[name+'Low']
        for col in ['SPY', '^IRX', '^TNX', '^VIX']:
            data[col + '_close'] = price[col+'Adj Close']
        # =============================================================================
        #             data[col+'_Open'] = price[('Open', col)]
        #             data[col+'_High'] = price[('High', col)]
        #             data[col+'_Low'] = price[('Low', col)]
        # =============================================================================
        feature_list += [data]
    
    
    print('run')
    for df in feature_list:
        col = df.columns[0][:3]
        if col == 'IGI':
            col = 'IGIB'
        temp = df[[col + '_close', col + '_volume', 'SPY_close']]
        for n in range(1, 5):
            for j in range(0,6):
                name = ['pre_returns' + '(%d,%d)' % (n, j), 'volume_change' + '(%d,%d)' % (n, j),
                        'SPY_pre_returns' + '(%d,%d)' % (n, j)]
                data = previous_returns(temp, n, j)
                df[name] = data
                #print(n, j)

    for df in feature_list:
        col = df.columns[0][:3]
        if col == 'IGI':
            col = 'IGIB'

        high, low, close, volume = df[col + '_High'], df[col + '_Low'], df[col + '_close'], df[col + '_volume']

        df[col + '_CCI'] = tl.CCI(high, low, close)
        df[col + '_RSI'] = tl.RSI(close)
        slowk, slowd = tl.STOCH(high, low, close)
        df[col + '_slowk'] = slowk
        df[col + '_slowd'] = slowd
        fastk, fastd = tl.STOCHF(high, low, close)
        df[col + '_fastk'] = slowk
        df[col + '_fastd'] = slowd
        df[col + '_WILLR'] = tl.WILLR(high, low, close)
        aroondown, aroonup = tl.AROON(high, low)
        df[col + '_aroondown'] = aroondown
        df[col + '_aroonup'] = aroonup
        # miss true strength index

        df[col + '_SMA'] = tl.SMA(close)
        df[col + '_EMA'] = tl.EMA(close)
        macd, macdsignal, macdhist = tl.MACD(close)
        df[col + '_macd'] = macd
        df[col + '_macdsignal'] = macdsignal
        df[col + '_macdhist'] = macdhist
        df[col + '_ADX'] = tl.ADX(high, low, close)
        # MACD,ADX are Momentum indicators
        df[col + '_T3'] = tl.T3(close)

        df[col + '_OBV'] = tl.OBV(close, volume)
        df[col + '_MFI'] = tl.MFI(high, low, close, volume)
        # MFI are Momentum indicators
        df[col + '_ADOSC'] = tl.ADOSC(high, low, close, volume)

        upperband, middleband, lowerband = tl.BBANDS(close)
        df[col + '_upperband'] = upperband
        df[col + '_middleband'] = middleband
        df[col + '_lowerband'] = lowerband
        df[col + '_ATR'] = tl.ATR(high, low, close)
        
        #df = df.shift(1)   # Move one period afterwards
        
        window = 3  ##wht 3?
        ret = df[col+'_returns'].shift(-1)
        y1 = np.zeros(len(ret))
        y1[ret<0] = -1
        y1[ret>0] = 1
        df['Y1'] = y1   
        
        # compute how much stock return change
        rolling_sd = ret.rolling(window=window).std()
        z2 = abs((ret - ret.shift(-1))/rolling_sd)
        y2 = np.zeros(len(ret))
        y2[z2<=1] = 1
        y2[z2>1] = 2
        df['Y2'] = y2
        
        
        df.dropna(axis=0, how='any', inplace=True)

    # feature selection
    for df in feature_list:
        col = df.columns[0][:3]
        if col == 'IGI':
            col = 'IGIB'

        df.drop([col + '_close', col + '_volume', col + '_returns', col + '_Open', col + '_High',
                 col + '_Low'], axis=1, inplace=True)




    train_date = list(feature_list[0].index)
    #train_date = input_index[:input_index.index(out_sample_begin)]
    feature_list_Y1 = copy.deepcopy(feature_list)
    feature_list_Y2 = copy.deepcopy(feature_list)
    dex = 0
    for df in feature_list:
        names = df.columns

        input_list = list(df.columns)
        input_list.remove('Y1')
        input_list.remove('Y2')

        input = df[input_list].loc[train_date]
        output = np.array(df[['Y1','Y2']].loc[train_date])
        #type = 'Random Forest'
        x_train, x_test, y_train, y_test = sk.model_selection.train_test_split(input, output, random_state=1,
                                                                               train_size=0.6)
        #print(x_train)
        scaler = sk.preprocessing.StandardScaler().fit(x_train)  
        x_train_transformed = scaler.transform(x_train)

        clf1 = RandomForestClassifier(n_estimators=100,max_features = 'sqrt',random_state=10)
        clf2 = RandomForestClassifier(n_estimators=100,max_features = 'sqrt',random_state=10)
        clf1.fit(x_train_transformed, y_train[:, 0])
        clf2.fit(x_train_transformed, y_train[:, 1])

        t1 = sorted(zip(map(lambda x: round(x, 4),  clf1.feature_importances_), names),
               reverse=True)
        f1_list = []
        #print(names)
        n_feature = 10
        for g in range(n_feature,len(names)-2):
            f1_list += [t1[g][1]]
        feature_list_Y1[dex].drop(f1_list, axis=1, inplace=True)
        feature_list_Y1[dex].drop('Y2', axis=1, inplace=True)

        t2 = sorted(zip(map(lambda x: round(x, 4),  clf2.feature_importances_), names),
               reverse=True)
        f2_list = []
        for g in range(n_feature,len(names)-2):
            f2_list += [t2[g][1]]

        feature_list_Y2[dex].drop(f2_list, axis=1, inplace=True)
        feature_list_Y2[dex].drop('Y1', axis=1, inplace=True)
        dex += 1
    return   feature_list_Y1, feature_list_Y2