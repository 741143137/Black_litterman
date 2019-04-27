#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 23:37:34 2019

@author: jax
"""

import talib as tl
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
pd.core.common.is_list_like = pd.api.types.is_list_like
import fix_yahoo_finance as yf
from pandas_datareader import data as pdr      # For download data


###################function####################
from sklearn.model_selection import cross_val_score
from sklearn import svm
import sklearn as sk
from sklearn.cross_validation import train_test_split, cross_val_score

from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

def accuracy(y1,y2,y1_t,y2_t):
    correct = 0
    for i in range(len(y1)):
        if (y1[i]==y1_t[i])&(y2[i]==y2_t[i]):
            correct += 1
    return correct/len(y1)

def cross_test(x_train_standard,y_train_standard,clf1,clf2):
    kf = sk.model_selection.KFold(n_splits=10)
    score = []
    for train, test in kf.split(x_train_standard):

        x_train = np.array(x_train_standard)[train,:]
        x_test = np.array(x_train_standard)[test,:]
        y_train =  np.array(y_train_standard)[train,:]
        y_test = np.array(y_train_standard)[test,:]
        scaler = sk.preprocessing.StandardScaler().fit(x_train)  # 通过训练集获得归一化函数模型。（也就是先减几，再除以几的函数）。在训练集和测试集上都使用这个归一化函数
        x_train_transformed = scaler.transform(x_train)
        x_test_transformed = scaler.transform(x_test)
    # clf = svm.SVC(C=0.1, kernel='linear', decision_function_shape='ovr')
        clf1 = svm.SVC(C=0.8, kernel='rbf', gamma=20, decision_function_shape='ovr')
        clf1.fit(x_train_transformed, y_train[:, 0].ravel())
        clf2 = svm.SVC(C=0.8, kernel='rbf', gamma=20, decision_function_shape='ovr')
        clf2.fit(x_train_transformed, y_train[:, 1].ravel())
        y1_hat_test = clf1.predict(x_test_transformed)
        y2_hat_test = clf2.predict(x_test_transformed)

        score += [accuracy(y1_hat_test,y2_hat_test,y_test[:,0] ,y_test[:,1])]
    return np.mean(score)

def classifier(x_train_transformed,x_test_transformed,y_train,y_test,type):

    if type == 'Bayesian':

        clf1 = GaussianNB()
        clf2 = GaussianNB()

        clf1.fit(x_train_transformed, y_train[:, 0])
        clf2.fit(x_train_transformed, y_train[:, 1])
        # print(clf1.score(x_train_transformed , y_train))   # 训练集正确率
        y1_hat_train = clf1.predict(x_train_transformed)
        y2_hat_train = clf2.predict(x_train_transformed)
        # print(clf1.score(x_test_transformed, y_test)) #测试集正确率
        y1_hat_test = clf1.predict(x_test_transformed)
        y2_hat_test = clf2.predict(x_test_transformed)
        train_accuracy =accuracy(y1_hat_train, y2_hat_train, y_train[:, 0], y_train[:, 1])
        test_accuracy =accuracy(y1_hat_test, y2_hat_test, y_test[:, 0], y_test[:, 1])

        x_train_standard = x_train
        y_train_standard = y_train
        scores = cross_test(x_train_standard, y_train_standard, clf1, clf2)


    elif type == 'Random Forest':
        clf1 = RandomForestClassifier(n_estimators=10, max_features=2)
        clf2 = RandomForestClassifier(n_estimators=10, max_features=2)
        clf1.fit(x_train_transformed, y_train[:, 0])
        clf2.fit(x_train_transformed, y_train[:, 1])
        # print(clf1.score(x_train_transformed , y_train))   # 训练集正确率
        y1_hat_train = clf1.predict(x_train_transformed)
        y2_hat_train = clf2.predict(x_train_transformed)
        # print(clf1.score(x_test_transformed, y_test)) #测试集正确率
        y1_hat_test = clf1.predict(x_test_transformed)
        y2_hat_test = clf2.predict(x_test_transformed)
        train_accuracy =accuracy(y1_hat_train, y2_hat_train, y_train[:, 0], y_train[:, 1])
        test_accuracy =accuracy(y1_hat_test, y2_hat_test, y_test[:, 0], y_test[:, 1])

        x_train_standard = x_train
        y_train_standard = y_train
        scores = cross_test(x_train_standard, y_train_standard, clf1, clf2)

    elif type == 'GB':
        clf1 = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
        clf2 = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
        clf1.fit(x_train_transformed, y_train[:, 0])
        clf2.fit(x_train_transformed, y_train[:, 1])
        # print(clf1.score(x_train_transformed , y_train))   # 训练集正确率
        y1_hat_train = clf1.predict(x_train_transformed)
        y2_hat_train = clf2.predict(x_train_transformed)
        # print(clf1.score(x_test_transformed, y_test)) #测试集正确率
        y1_hat_test = clf1.predict(x_test_transformed)
        y2_hat_test = clf2.predict(x_test_transformed)
        train_accuracy =accuracy(y1_hat_train, y2_hat_train, y_train[:, 0], y_train[:, 1])
        test_accuracy =accuracy(y1_hat_test, y2_hat_test, y_test[:, 0], y_test[:, 1])

        x_train_standard = x_train
        y_train_standard = y_train
        scores = cross_test(x_train_standard, y_train_standard, clf1, clf2)


    elif type == 'Adaboosting':
        clf1 = AdaBoostClassifier(n_estimators=100)
        clf2 = AdaBoostClassifier(n_estimators=100)
        clf1.fit(x_train_transformed, y_train[:, 0])
        clf2.fit(x_train_transformed, y_train[:, 1])
        # print(clf1.score(x_train_transformed , y_train))   # 训练集正确率
        y1_hat_train = clf1.predict(x_train_transformed)
        y2_hat_train = clf2.predict(x_train_transformed)
        # print(clf1.score(x_test_transformed, y_test)) #测试集正确率
        y1_hat_test = clf1.predict(x_test_transformed)
        y2_hat_test = clf2.predict(x_test_transformed)
        train_accuracy =accuracy(y1_hat_train, y2_hat_train, y_train[:, 0], y_train[:, 1])
        test_accuracy =accuracy(y1_hat_test, y2_hat_test, y_test[:, 0], y_test[:, 1])

        x_train_standard = x_train
        y_train_standard = y_train
        scores = cross_test(x_train_standard, y_train_standard, clf1, clf2)


    elif type == 'Decision Tree':
        clf1 = DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=7,
                                     max_features=None, max_leaf_nodes=None,
                                     min_samples_leaf=10,
                                     min_samples_split=20, min_weight_fraction_leaf=0.0,
                                     presort=False, random_state=None, splitter='best')
        clf2 = DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=7,
                                     max_features=None, max_leaf_nodes=None,
                                     min_samples_leaf=10,
                                     min_samples_split=20, min_weight_fraction_leaf=0.0,
                                     presort=False, random_state=None, splitter='best')
        # random state?
        clf1.fit(x_train_transformed, y_train[:, 0])
        clf2.fit(x_train_transformed, y_train[:, 1])
        # print(clf1.score(x_train_transformed , y_train))   # 训练集正确率
        y1_hat_train = clf1.predict(x_train_transformed)
        y2_hat_train = clf2.predict(x_train_transformed)
        # print(clf1.score(x_test_transformed, y_test)) #测试集正确率
        y1_hat_test = clf1.predict(x_test_transformed)
        y2_hat_test = clf2.predict(x_test_transformed)
        train_accuracy =accuracy(y1_hat_train, y2_hat_train, y_train[:, 0], y_train[:, 1])
        test_accuracy =accuracy(y1_hat_test, y2_hat_test, y_test[:, 0], y_test[:, 1])
        x_train_standard = x_train
        y_train_standard = y_train
        scores = cross_test(x_train_standard, y_train_standard, clf1, clf2)


    elif type == 'KNN':
        clf1 = KNeighborsClassifier(n_neighbors=5)
        clf2 = KNeighborsClassifier(n_neighbors=5)
        clf1.fit(x_train_transformed, y_train[:, 0])
        clf2.fit(x_train_transformed, y_train[:, 1])
        # print(clf1.score(x_train_transformed , y_train))   # 训练集正确率
        y1_hat_train = clf1.predict(x_train_transformed)
        y2_hat_train = clf2.predict(x_train_transformed)
        # print(clf1.score(x_test_transformed, y_test)) #测试集正确率
        y1_hat_test = clf1.predict(x_test_transformed)
        y2_hat_test = clf2.predict(x_test_transformed)
        train_accuracy =accuracy(y1_hat_train, y2_hat_train, y_train[:, 0], y_train[:, 1])
        test_accuracy =accuracy(y1_hat_test, y2_hat_test, y_test[:, 0], y_test[:, 1])

        x_train_standard = x_train
        y_train_standard = y_train
        scores = cross_test(x_train_standard, y_train_standard, clf1, clf2)


    elif type == 'Logistic':
        clf1 = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(x_train_transformed,y_train)
        clf2 = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(x_train_transformed,y_train)
        clf1.fit(x_train_transformed, y_train[:, 0])
        clf2.fit(x_train_transformed, y_train[:, 1])
        # print(clf1.score(x_train_transformed , y_train))   # 训练集正确率
        y1_hat_train = clf1.predict(x_train_transformed)
        y2_hat_train = clf2.predict(x_train_transformed)
        # print(clf1.score(x_test_transformed, y_test)) #测试集正确率
        y1_hat_test = clf1.predict(x_test_transformed)
        y2_hat_test = clf2.predict(x_test_transformed)
        train_accuracy =accuracy(y1_hat_train, y2_hat_train, y_train[:, 0], y_train[:, 1])
        test_accuracy =accuracy(y1_hat_test, y2_hat_test, y_test[:, 0], y_test[:, 1])

        x_train_standard = x_train
        y_train_standard = y_train
        scores = cross_test(x_train_standard, y_train_standard, clf1, clf2)


    elif type == 'SVM':
        clf1 = svm.SVC(C=0.8, kernel='rbf', gamma=20, decision_function_shape='ovr')
        clf2 = svm.SVC(C=0.8, kernel='rbf', gamma=20, decision_function_shape='ovr')
        # clf = svm.SVC(C=0.8, kernel='rbf', gamma=20, decision_function_shape='ovo')
        clf1.fit(x_train_transformed, y_train[:, 0])
        clf2.fit(x_train_transformed, y_train[:, 1])
        # print(clf1.score(x_train_transformed , y_train))   # 训练集正确率
        y1_hat_train = clf1.predict(x_train_transformed)
        y2_hat_train = clf2.predict(x_train_transformed)
        # print(clf1.score(x_test_transformed, y_test)) #测试集正确率
        y1_hat_test = clf1.predict(x_test_transformed)
        y2_hat_test = clf2.predict(x_test_transformed)
        train_accuracy =accuracy(y1_hat_train, y2_hat_train, y_train[:, 0], y_train[:, 1])
        test_accuracy = accuracy(y1_hat_test, y2_hat_test, y_test[:, 0], y_test[:, 1])

        x_train_standard = x_train
        y_train_standard = y_train
        scores = cross_test(x_train_standard, y_train_standard, clf1, clf2)


    return clf1,clf2 ,[train_accuracy,test_accuracy,scores]



def previous_returns(df, n, j):
    #n weeks, j lags
    returns = df.pct_change(periods = 5*n)
    returns = returns.shift(j)

    return returns

def feature_selection(df):
    Y1 = df['Y1']
    Y2 = df['Y2']
    featurename_list = list(df.columns)
    featurename_list.remove('Y1')
    featurename_list.remove('Y2')
    
    drop_list = []
    for featurename in featurename_list:
        A = df[featurename][df['Y1'] == np.unique(Y1)[0]]
        B = df[featurename][df['Y1'] == np.unique(Y1)[1]]
        SE = np.sqrt(np.var(A)/len(A) + np.var(B)/len(B))
        score = abs(np.mean(A) - np.mean(B)) / SE
        
        A2 = df[featurename][df['Y2'] == np.unique(Y2)[0]]
        B2 = df[featurename][df['Y2'] == np.unique(Y2)[1]] 
        SE2 = np.sqrt(np.var(A2)/len(A2) + np.var(B2)/len(B2))        
        score2 = abs(np.mean(A2) - np.mean(B2)) / SE2
        if score <= 1 or score2 <= 1:
            drop_list += [featurename]

    return drop_list



            
        
        



if __name__ == '__main__':
    yf.pdr_override()
    tickers = ['EMB','GLD','TLT','IYR','IGIB','IJH']
    features = ['SPY','^VIX','^TNX', '^IRX']
    
    st = datetime.datetime(2008,11,19)
    end = datetime.datetime(2019,4,18)
    close = pdr.get_data_yahoo(tickers,start = st,end = end)["Adj Close"]
    close2 = pdr.get_data_yahoo(features,start = st,end = end)["Adj Close"]
    price = pdr.get_data_yahoo(tickers + features,start = st,end = end)
    
#    Volumn = pdr.get_data_yahoo(['EMB','GLD','TLT','IYR','IGIB','IJH'],start = st,end = end)["Volume"]
    
    returns = close.pct_change()
    
    feature_list = []
    for name in close.columns:
        data = pd.DataFrame(close[name])
        data.columns = [name+'_close']
        data[name+'_volume'] = price[('Volume', name)]
        data[name+'_returns'] = returns[name]
        data[name+'_Open'] = price[('Open', name)]
        data[name+'_High'] = price[('High', name)]
        data[name+'_Low'] = price[('Low', name)]
        for col in ['SPY', '^IRX', '^TNX', '^VIX']:
            data[col+'_close'] = price[('Adj Close', col)]    
# =============================================================================
#             data[col+'_Open'] = price[('Open', col)]
#             data[col+'_High'] = price[('High', col)]
#             data[col+'_Low'] = price[('Low', col)]
# =============================================================================
        feature_list += [data]
                
    for df in feature_list:
        col = df.columns[0][:3]
        if col == 'IGI':
            col = 'IGIB'
        temp = df[[col+'_close', col+'_volume', 'SPY_close']]
        for n in range(1,5):
            for j in range(6):
                name = ['pre_returns' + '(%d,%d)' % (n,j), 'volume_change' + '(%d,%d)' % (n,j), 
                        'SPY_pre_returns' + '(%d,%d)' % (n,j)]
                data = previous_returns(temp, n, j)
                df[name] = data
                print(n,j)

    #CCI(high, low, close, timeperiod=14)   
    #RSI(close, timeperiod=14)   
    #STOCH(high, low, close, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)  
    #STOCHF(high, low, close, fastk_period=5, fastd_period=3, fastd_matype=0)
    #WILLR(high, low, close, timeperiod=14)       
    #AROON(high, low, timeperiod=14)
    
    #SMA(close, timeperiod=30)
    #EMA(close, timeperiod=30)
    #MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    #ADX(high, low, close, timeperiod=14)
    #T3(close, timeperiod=5, vfactor=0)
    
    #OBV(close, volume)
    #MFI(high, low, close, volume, timeperiod=14)
    #no CMF,Force indica, add ADOSC instead
    #ADOSC(high, low, close, volume, fastperiod=3, slowperiod=10)
    
    #BBANDS(close, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)
    #ATR(high, low, close, timeperiod=14)
    #no VR, HHLL
    for df in feature_list:
        col = df.columns[0][:3]
        if col == 'IGI':
            col = 'IGIB'

        high, low, close, volume = df[col+'_High'], df[col+'_Low'], df[col+'_close'], df[col+'_volume']
        df[col+'_CCI'] = tl.CCI(high, low, close)
        df[col+'_RSI'] = tl.RSI(close)
        slowk, slowd = tl.STOCH(high, low, close)
        df[col+'_slowk'] = slowk
        df[col+'_slowd'] = slowd
        fastk, fastd = tl.STOCHF(high, low, close)
        df[col+'_fastk'] = slowk
        df[col+'_fastd'] = slowd
        df[col+'_WILLR'] = tl.WILLR(high, low, close)
        aroondown, aroonup = tl.AROON(high, low)
        df[col+'_aroondown'] = aroondown
        df[col+'_aroonup'] = aroonup
        #miss true strength index
        
        
        df[col+'_SMA'] = tl.SMA(close)
        df[col+'_EMA'] = tl.EMA(close)
        macd, macdsignal, macdhist = tl.MACD(close)
        df[col+'_macd'] = macd
        df[col+'_macdsignal'] = macdsignal
        df[col+'_macdhist'] = macdhist
        df[col+'_ADX'] = tl.ADX(high, low, close)
        #MACD,ADX are Momentum indicators
        df[col+'_T3'] = tl.T3(close)
        
               
        df[col+'_OBV'] = tl.OBV(close, volume)
        df[col+'_MFI'] = tl.MFI(high, low, close, volume)
        #MFI are Momentum indicators
        df[col+'_ADOSC'] = tl.ADOSC(high, low, close, volume)
        
        
        upperband, middleband, lowerband = tl.BBANDS(close)
        df[col+'_upperband'] = upperband
        df[col+'_middleband'] = middleband
        df[col+'_lowerband'] = lowerband       
        df[col+'_ATR'] = tl.ATR(high, low, close)
        
               
        window = 3
        ret = df[col+'_returns']
        df['Y1'] = (ret < 0).replace(True,1)   
        df['Y1'] = df['Y1'].replace(0.0,-1) 
        # compute how much stock return change
        rolling_sd = ret.rolling(window=window).std()
        z2 = abs((ret - ret.shift(-1))/rolling_sd)
        df['Y2'] = (z2 <= 1).replace(False,2)
       
        df.dropna(axis=0, how='any', inplace=True)


    #feature selection
    for df in feature_list:
        col = df.columns[0][:3]
        if col == 'IGI':
            col = 'IGIB'
        drop_list = feature_selection(df.iloc[:,6:])
        df.drop(drop_list, axis=1, inplace = True)
        df.drop([col+'_close', col+'_volume', col+'_returns', col+'_Open', col+'_High', 
                 col+'_Low'], axis=1, inplace = True)