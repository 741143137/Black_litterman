#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 23:31:04 2019

@author: jax
"""

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 09:20:50 2019

@author: jax
"""

# -------------------------------------------- Get Data & Parameters ------------------------------------------------

"""
1.0 Initialization
"""
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt

#pd.core.common.is_list_like = pd.api.types.is_list_like
#import fix_yahoo_finance as yf
#from pandas_datareader import data as pdr  # For download data

# Own code function
# from Backtest import Backtest
# from feature_selection_1 import feature_intro, get_view
#from feature_selection_final import feature_intro, get_view
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
import talib as tl
import pandas as pd


def classifier(x_train_transformed,x_test_transformed,y_train,y_test,type):

    if type == 'Bayesian':

        clf = GaussianNB()

        clf.fit(x_train_transformed, y_train)
        train_accuracy = clf.score(x_train_transformed, y_train)
        test_accuracy = clf.score(x_test_transformed, y_test)
        scores = np.mean(cross_val_score(clf, x_train_transformed, y_train))

    elif type == 'Random Forest':
        clf = RandomForestClassifier(n_estimators=20)
        clf.fit(x_train_transformed, y_train)
        train_accuracy = clf.score(x_train_transformed, y_train)
        test_accuracy =clf.score(x_test_transformed, y_test)
        scores = np.mean(cross_val_score(clf, x_train_transformed, y_train))

    elif type == 'GB':
        clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
        clf.fit(x_train_transformed, y_train)
        train_accuracy = clf.score(x_train_transformed, y_train)
        test_accuracy =clf.score(x_test_transformed, y_test)
        scores = np.mean(cross_val_score(clf, x_train_transformed, y_train))

    elif type == 'Adaboosting':
        clf = AdaBoostClassifier(n_estimators=100)
        clf.fit(x_train_transformed, y_train)
        train_accuracy = clf.score(x_train_transformed, y_train)
        test_accuracy =clf.score(x_test_transformed, y_test)
        scores = np.mean(cross_val_score(clf, x_train_transformed, y_train))

    elif type == 'Decision Tree':
        clf = DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=7,
                                     max_features=None, max_leaf_nodes=None,
                                     min_samples_leaf=10,
                                     min_samples_split=20, min_weight_fraction_leaf=0.0,
                                     presort=False, random_state=None, splitter='best')
        # random state?
        clf.fit(x_train_transformed, y_train)
        train_accuracy = clf.score(x_train_transformed, y_train)
        test_accuracy =clf.score(x_test_transformed, y_test)
        scores = np.mean(cross_val_score(clf, x_train_transformed, y_train))

    elif type == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=5)
        clf.fit(x_train_transformed, y_train)
        train_accuracy = clf.score(x_train_transformed, y_train)
        test_accuracy = clf.score(x_test_transformed, y_test)
        scores = np.mean(cross_val_score(clf, x_train_transformed, y_train, cv=5))

    elif type == 'Logistic':
        clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')
        clf.fit(x_train_transformed, y_train)
        train_accuracy = clf.score(x_train_transformed, y_train)
        test_accuracy =clf.score(x_test_transformed, y_test)
        scores = np.mean(cross_val_score(clf, x_train_transformed, y_train, cv=5))

    elif type == 'SVM':
        clf = svm.SVC(C=0.8, kernel='rbf', gamma=20, decision_function_shape='ovr')
        # clf = svm.SVC(C=0.8, kernel='rbf', gamma=20, decision_function_shape='ovo')
        clf.fit(x_train_transformed, y_train.ravel())
        train_accuracy = clf.score(x_train_transformed, y_train)  # 训练集正确率
        test_accuracy = clf.score(x_test_transformed, y_test)  # 测试集正确率
        scores = np.mean(cross_val_score(clf, x_train_transformed, y_train, cv=5))


    return clf ,[train_accuracy,test_accuracy,scores]



def previous_returns(df, n, j):
    #n weeks, j lags
    #returns = df.pct_change(periods = 5*n)
    returns = df.pct_change(periods = n)
    returns = returns.shift(j)

    return returns


def feature_selection(df):
    Y = df['Y']
    featurename_list = list(df.columns)
    featurename_list.remove('Y')

    drop_list = []
    for featurename in featurename_list:
        A = df[featurename][df['Y'] == np.unique(Y)[0]]
        B = df[featurename][df['Y'] == np.unique(Y)[1]]
        SE = np.sqrt(np.var(A) / len(A) + np.var(B) / len(B))
        score = abs(np.mean(A) - np.mean(B)) / SE

        if score <= 1:
            drop_list += [featurename]

    return drop_list


#feature means that the next week return at this time point using weekly data
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
    #print(feature_list)
    # CCI(high, low, close, timeperiod=14)
    # RSI(close, timeperiod=14)
    # STOCH(high, low, close, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
    # STOCHF(high, low, close, fastk_period=5, fastd_period=3, fastd_matype=0)
    # WILLR(high, low, close, timeperiod=14)
    # AROON(high, low, timeperiod=14)

    # SMA(close, timeperiod=30)
    # EMA(close, timeperiod=30)
    # MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    # ADX(high, low, close, timeperiod=14)
    # T3(close, timeperiod=5, vfactor=0)

    # OBV(close, volume)
    # MFI(high, low, close, volume, timeperiod=14)
    # no CMF,Force indica, add ADOSC instead
    # ADOSC(high, low, close, volume, fastperiod=3, slowperiod=10)

    # BBANDS(close, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)
    # ATR(high, low, close, timeperiod=14)
    # no VR, HHLL
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

        window = 3  ##wht 3?
        ret = df[col + '_returns'].shift(-1)
        Y1 = (ret < 0).replace(True, 1)
        Y1 = Y1.replace(0.0, -1)
        # compute how much stock return change
        rolling_sd = ret.rolling(window=window).std()
        rolling_mean = ret.rolling(window=window).mean()
        z2 = abs((ret - rolling_mean) / rolling_sd)
        Y2 = (z2 <= 1).replace(False, 2)
        
        Y1 = np.zeros(len(ret))
        Y1[ret<0] = -1
        Y1[ret>0] = 1 
        
        # compute how much stock return change
        rolling_sd = ret.rolling(window=window).std()
        z2 = abs((ret - ret.shift(-1))/rolling_sd)
        Y2 = np.zeros(len(ret))
        Y2[z2<=1] = 1
        Y2[z2>1] = 2
        df['Y'] = 0

        datelist = df.index
        for i in range(len(df['Y'])):
            if (Y1[i] == -1) & (Y2[i] == 2):
                df.loc[datelist[i], 'Y'] = 0
            if (Y1[i] == -1) & (Y2[i] == 1):
                df.loc[datelist[i], 'Y'] = 1
            if (Y1[i] == 1) & (Y2[i] == 1):
                df.loc[datelist[i], 'Y'] = 2
            if (Y1[i] == 1) & (Y2[i] == 2):
                df.loc[datelist[i], 'Y'] = 3

        df.dropna(axis=0, how='any', inplace=True)

    # feature selection
    for df in feature_list:
        col = df.columns[0][:3]
        if col == 'IGI':
            col = 'IGIB'
#        drop_list = feature_selection(df.iloc[:, 6:])
#        df.drop(drop_list, axis=1, inplace=True)
        df.drop([col + '_close', col + '_volume', col + '_returns', col + '_Open', col + '_High',
                 col + '_Low'], axis=1, inplace=True)

    train_date = list(feature_list[0].index)
    #train_date = input_index[:input_index.index(out_sample_begin)]
    for df in feature_list:
        names = df.columns
        input_list = list(df.columns)
        input_list.remove('Y')

        input = df[input_list].loc[train_date]
        output = np.array(df['Y'].loc[train_date])
        #type = 'Random Forest'
        x_train, x_test, y_train, y_test = sk.model_selection.train_test_split(input, output, random_state=1,
                                                                               train_size=0.6)
        #print(x_train)
        scaler = sk.preprocessing.StandardScaler().fit(x_train)  # 通过训练集获得归一化函数模型。（也就是先减几，再除以几的函数）。在训练集和测试集上都使用这个归一化函数
        x_train_transformed = scaler.transform(x_train)
        x_test_transformed = scaler.transform(x_test)
        clf = RandomForestClassifier(n_estimators=10, max_features=2)
        clf.fit(x_train_transformed, y_train)
        t = sorted(zip(map(lambda x: round(x, 4),  clf.feature_importances_), names),
               reverse=True)
        f_list = []
        for g in range(10,len(names)-1):
            f_list += [t[g][1]]

        df.drop(f_list, axis=1, inplace=True)

    return feature_list

    # input = features

T = 0
def get_view(feature_list, type, out_sample_begin):
    global T
    ml_model = {}
    i = 0
    info = []
    input_index = list(feature_list[0].index)
    tickers = ['EMB','GLD','TLT','IYR','IGIB','IJH']
    train_date = input_index[:input_index.index(out_sample_begin)]
#    out_sample_date = input_index[input_index.index(out_sample_begin):]
    #train_date = input_index[input_index.index(out_sample_begin)-200:input_index.index(out_sample_begin)]
    out_sample_date = input_index[input_index.index(out_sample_begin):]
    # input = df[input_list][train_date]


    for df in feature_list:
        input_list = list(df.columns)
        input_list.remove('Y')

        input = df[input_list].loc[train_date]
        output = np.array(df['Y'].loc[train_date])

        x_train, x_test, y_train, y_test = sk.model_selection.train_test_split(input, output, random_state=1,
                                                                               train_size=0.6)

        scaler = sk.preprocessing.StandardScaler().fit(x_train)  # 通过训练集获得归一化函数模型。（也就是先减几，再除以几的函数）。在训练集和测试集上都使用这个归一化函数
        x_train_transformed = scaler.transform(x_train)
        x_test_transformed = scaler.transform(x_test)
        clf, result = classifier(x_train_transformed, x_test_transformed, y_train, y_test, type)

        ml_model['%s' % tickers[i]] = {}
        ml_model[tickers[i]]['model'] = clf
        ml_model[tickers[i]]['accuracy'] = result
        ml_model[tickers[i]]['transform'] = scaler
        i += 1

        info += [df[input_list].loc[out_sample_date]]

    view = []
    #for j in range(len(info)):
    #    y1 = ml_model[tickers[j]]['model1'].predict(info[j])
    #    y2 = ml_model[tickers[j]]['model2'].predict(info[j])
    #    y_tran = y1 * y2
    #    view += [y_tran]
    #view = np.matrix(view)
    #view = pd.DataFrame(view, index=out_sample_date, columns=tickers)

    j = 0
    for df in feature_list:
        input_list = list(df.columns)
        input_list.remove('Y')
        input = df[input_list]

        input_transform = ml_model[tickers[j]]['transform'].transform(input.loc[out_sample_date])
        y_pre = ml_model[tickers[j]]['model'].predict(input_transform)
        #print(y_pre)
        y_pre[y_pre==0] = -2
        y_pre[y_pre == 1] = -1
        y_pre[y_pre == 2] = 1
        y_pre[y_pre == 3] = 2
        #print(y_pre)
        print(ml_model[tickers[j]]['model'].score(input_transform,np.array(df['Y'].loc[out_sample_date])))
        view += [y_pre]
        j += 1
    view = np.matrix(view).T

    #view = pd.DataFrame(view, index=out_sample_date, columns=tickers)
    view = pd.DataFrame(view, index=out_sample_date , columns=tickers)
    #view_final = view.loc[out_sample_begin]
    return view, ml_model

#yf.pdr_override()
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
market_weights = pd.read_csv('market_weights.csv', index_col=0)

# =========================== Data processing =================================================
# window = 3
# N = len(ret)
# M = len(tickers)
# y = np.zeros([N-3,M])
# rolling_sd = ret.rolling(window=window).std()
# z2 = abs((ret - ret.shift(-1))/rolling_sd)
# y1 = np.zeros([N,M])
# y1[ret<0] = -1
# y1[ret>0] = 1
#
## compute how much stock return change
# y2 = np.zeros([N,M])
# y2[z2<=1] = 1
# y2[z2>1] = 2

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

###########################time choose##########################################

st = datetime.datetime(2012,1,5)
end = datetime.datetime(2017, 12, 28)

price = price.iloc[date.index(st):(date.index(end)+1)]
close = close.iloc[date.index(st):(date.index(end)+1)]
ret= ret.iloc[date.index(st):(date.index(end))]

feature_list = feature_intro(close, price)

# ============================= Train Data using Classification==============================


# ===================================================
''' parameter initialziation '''
N_length = 250
window_rebalance = 1  # Rebalance every week
labda = 0.94
tao = 1 / (N_length / 250)  # Following result of Meucci(2010)
# 7 year market risk aversion, can try points like 0.2, 0.5,1,2
risk_aversion = 0.5344

total_ret = np.zeros(len(ret))

test_pi = []
day_list = list(ret.index)

out_sample_begin = pd.to_datetime( ret.index[day_list.index(datetime.datetime(2016,1,7))], format="%Y-%m-%d")
type = 'Random Forest' #'random forest' #'SVM'  'Bayesian' 'KNN'
view_final, ml_model = get_view(feature_list, type, out_sample_begin)

model_1 = ml_model

################################model storage###################################################
#from sklearn.externals import joblib
#for i in range(len(tickers)):
#    joblib.dump(ml_model[tickers[i]]['model'], tickers[i] + '.pkl')

#ml_model = {}
#for i in range(len(tickers)):
    #joblib.dump(ml_model[tickers[i]]['model'], tickers[i] + '.pkl')
#    ml_model[tickers[i]]['model'] = joblib.load(tickers[i] + '.pkl')
###########################################################################################

weight_final = []
date_final = []
for i in range(day_list.index(out_sample_begin), len(ret), window_rebalance):

    # There are two different ways to estimate current cov
    # 1 is use sample cov, 2 is use EWMA covariance matrix
    week_list = list(ret.index)
    #out_sample_at = pd.to_datetime( ret.index[i], format="%Y-%m-%d")
    out_sample_at = pd.to_datetime(week_list[i], format="%Y-%m-%d")
    cov_matrix = ret.iloc[:i, :].cov()

    last_day_return = np.matrix(ret.iloc[week_list.index(out_sample_at)] - ret.iloc[week_list.index(out_sample_at)].mean()).T
    #print( last_day_return)
    current_cov = np.matrix(labda * cov_matrix + (1 - labda) * last_day_return * last_day_return.T)

    # Obtain market capital weight
    capital_weight = get_market_weight(ret.index[i])
    # Equivebriant return
    pi = np.matrix(risk_aversion * current_cov * capital_weight)

    ''' test 
    test_pi.append(pi)'''

    # Get subjective view by machine learning classification
    #print(ret.index[i])
    #out_sample_begin = ret.index[i].to_datetime()# datetime.datetime(2017,2,13)

    #print(out_sample_begin )
    date_final += [out_sample_at]
    # multi_class
    view = view_final.loc[out_sample_at]
    #view = 0
    #view = get_view(feature_list, type, ret.index[250])

    individual_var = np.diagonal(current_cov)
    Q = pi + np.matrix(view * np.array(individual_var ** 0.5) * 0.2).T

    # Note, since we
    omega = np.matrix(np.diag(individual_var))

    posterior_mean = pi + tao * current_cov * (current_cov * tao + omega).I * (Q - pi)
    # Exact formula should be pi + tao* np.dot(current_cov.I,Q - pi)
    posterior_sigma = current_cov + ((tao * current_cov).I + omega.I).I

    optimal_weight = posterior_sigma.I * posterior_mean / risk_aversion
    optimal_weight = optimal_weight / optimal_weight.sum()

    if i == day_list.index(out_sample_begin):
        weight_final =optimal_weight
    else:
        weight_final = np.hstack((weight_final, optimal_weight))
    # calculate return

    if i + window_rebalance < len(ret):
        total_ret[i:i + window_rebalance] = np.dot(ret.iloc[i:i + window_rebalance, :], optimal_weight).getA()[:, 0]
    else:
        total_ret[i:] = np.dot(ret.iloc[i:, :], optimal_weight).getA()[:, 0]

weight_final_1 = np.array(weight_final).T
weight_list = pd.DataFrame(weight_final_1,index = date_final,columns = tickers)

final_ret = pd.Series(total_ret, index=ret.index).dropna()
net_value = np.cumprod(1 + final_ret[final_ret != 0])
plt.plot(net_value)


net_value1 = net_value
plt.plot(net_value1)
a = np.array(final_ret[final_ret != 0])
# backtest = Backtest(a,'Market Index',0.05)
# print(backtest.summary())
