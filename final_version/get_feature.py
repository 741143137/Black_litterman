#!/usr/bin/env python3
# -*- coding: utf-8 -*-


'''
Get_feature.py includes feature engineering and feature selection for each ETF

We gather features from six information sets: Time series data for return, Time series data for volume change,
Momentum related technical indicators (like RSI), Trend related technical indicators (like MACD),
Volume based technical indicators (like OBV), Volatility based indicators (like Bollinger Bands)

We used to feature importance to accomplish feature selection. Specifically for this classification problem, we built
random forest model, and used Gini index as feature importance.

(Note, get_feature part is done by my teammates)
'''


import pandas as pd
import copy
import numpy as np

import sklearn as sk
from sklearn.ensemble import RandomForestClassifier
''' important '''
# Need to pip install ta-lib library for feature engineering part
# If you have problems about download this library, please check this website:
#    https://mrjbq7.github.io/ta-lib/install.html 
import talib as tl


# Previous_returns is the function to get lag terms of time series data

def previous_returns(df, n, j):
    '''
    
    Arguments:
    df -- Dataframe of specific features
    n -- n weeks of moving window
    j -- j lag tern
    
    Returns:
    the dataframe of lag term
    '''
 
    returns = df.pct_change(periods = n)
    returns = returns.shift(j)

    return returns


# feature_into compute total 105 features for six ETFs and use feature importance to do feature selection

def feature_intro(close, price_info):
    ''' 
    
    Arguments:
    close -- the close price DataFrame for six ETF, where row is time series, column is ETF name
    price_info_info -- DataFrame includes price information like open,high,low for ETF and other features like SPY, VIX
    
    Returns:
    feature_list_Y1 -- A list of Dataframe includes seleted features for each ETF, Y1 means the label is 
                       direction of return
    feature_list_Y2 -- A list of Dataframe includes seleted features for each ETF, Y2 means the label is 
                       whether the return change a lot or not (measure by one standard deviation away from n day mean)
    '''
    
    # ==========================  feature engineering  ===============================
    returns = close.pct_change()
    feature_list = []   # Initialization
    
    # Iterate each ETF to compute their own features
    for name in close.columns:
        data = pd.DataFrame(close[name])
        data.columns = [name + '_close']
        data[name + '_volume'] = price_info[ name+'Volume']
        data[name + '_returns'] = returns[name]
        data[name + '_Open'] = price_info[name+'Open']
        data[name + '_High'] = price_info[name+'High']
        data[name + '_Low'] = price_info[name+'Low']
        for col in ['SPY', '^IRX', '^TNX', '^VIX']:
            data[col + '_close'] = price_info[col+'Adj Close']

        feature_list += [data]
    
    
    # Add lag term of return and volume change into features
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

    # Add technical indicators into features
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
        
        # Create label for the first classification problem: the direction of return (increase or decrease)
        window = 3  
        ret = df[col+'_returns'].shift(-1)
        y1 = np.zeros(len(ret))
        y1[ret<0] = -1
        y1[ret>0] = 1
        df['Y1'] = y1   
        
        # compute how much stock return change (whether is one standard deviation away from mean or not)
        rolling_sd = ret.rolling(window=window).std()
        z2 = abs((ret - ret.shift(-1))/rolling_sd)
        y2 = np.zeros(len(ret))
        y2[z2<=1] = 1
        y2[z2>1] = 2
        df['Y2'] = y2
        
        # Drop na value       
        df.dropna(axis=0, how='any', inplace=True)

    for df in feature_list:
        col = df.columns[0][:3]
        if col == 'IGI':
            col = 'IGIB'

        df.drop([col + '_close', col + '_volume', col + '_returns', col + '_Open', col + '_High',
                 col + '_Low'], axis=1, inplace=True)

    #  =================  Use random forest to select features ======================================
    
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
        
        # Split the data into train set and test set
        x_train, x_test, y_train, y_test = sk.model_selection.train_test_split(input, output, random_state=1,
                                                                               train_size=0.6)
        # Standardize features
        scaler = sk.preprocessing.StandardScaler().fit(x_train)  
        x_train_transformed = scaler.transform(x_train)
        
        # Built random forest model for two classification problems
        clf1 = RandomForestClassifier(n_estimators=100,max_features = 'sqrt',random_state=10)
        clf2 = RandomForestClassifier(n_estimators=100,max_features = 'sqrt',random_state=10)
        clf1.fit(x_train_transformed, y_train[:, 0])
        clf2.fit(x_train_transformed, y_train[:, 1])
        
        # Rank the features accoridng to feature importance
        t1 = sorted(zip(map(lambda x: round(x, 4),  clf1.feature_importances_), names),
               reverse=True)
        f1_list = []
        
        
        # Choose top 10 features for each classification problem
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
