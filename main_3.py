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
import copy
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


def accuracy(y1,y2,y1_t,y2_t):
    correct = 0
    for i in range(len(y1)):
        if (y1[i]==y1_t[i])&(y2[i]==y2_t[i]):
            correct += 1
    return correct/len(y1)

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
        scores = np.mean(cross_val_score(clf, x_train, y_train))

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
    Y1 = df['Y1']
    Y2 = df['Y2']
    featurename_list = list(df.columns)
    featurename_list.remove('Y1')
    featurename_list.remove('Y2')

    drop_list = []
    for featurename in featurename_list:
        A = df[featurename][df['Y1'] == np.unique(Y1)[0]]
        B = df[featurename][df['Y1'] == np.unique(Y1)[1]]
        SE = np.sqrt(np.var(A) / len(A) + np.var(B) / len(B))
        score = abs(np.mean(A) - np.mean(B)) / SE

        A2 = df[featurename][df['Y2'] == np.unique(Y2)[0]]
        B2 = df[featurename][df['Y2'] == np.unique(Y2)[1]]
        SE2 = np.sqrt(np.var(A2) / len(A2) + np.var(B2) / len(B2))
        score2 = abs(np.mean(A2) - np.mean(B2)) / SE2
        if score <= 1 or score2 <= 1:
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
        ret = df[col+'_returns']
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
#        drop_list = feature_selection(df.iloc[:, 6:])
#        df.drop(drop_list, axis=1, inplace=True)
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
        scaler = sk.preprocessing.StandardScaler().fit(x_train)  # 通过训练集获得归一化函数模型。（也就是先减几，再除以几的函数）。在训练集和测试集上都使用这个归一化函数
        x_train_transformed = scaler.transform(x_train)

        clf1 = RandomForestClassifier(n_estimators=10)
        clf2 = RandomForestClassifier(n_estimators=10)
        clf1.fit(x_train_transformed, y_train[:, 0])
        clf2.fit(x_train_transformed, y_train[:, 1])

        t1 = sorted(zip(map(lambda x: round(x, 4),  clf1.feature_importances_), names),
               reverse=True)
        f1_list = []
        #print(names)
        for g in range(10,len(names)-2):
            f1_list += [t1[g][1]]
        feature_list_Y1[dex].drop(f1_list, axis=1, inplace=True)
        feature_list_Y1[dex].drop('Y2', axis=1, inplace=True)

        t2 = sorted(zip(map(lambda x: round(x, 4),  clf2.feature_importances_), names),
               reverse=True)
        f2_list = []
        for g in range(10,len(names)-2):
            f2_list += [t2[g][1]]

        feature_list_Y2[dex].drop(f2_list, axis=1, inplace=True)
        feature_list_Y2[dex].drop('Y1', axis=1, inplace=True)
        dex += 1
    return   feature_list_Y1, feature_list_Y2

    # input = features


T = 0
def get_view(feature_list_Y1, feature_list_Y2, type, out_sample_begin):
    global T
    ml_model = {}
    i = 0

    input_index_Y1 = list(feature_list_Y1[0].index)
    tickers = ['EMB','GLD','TLT','IYR','IGIB','IJH']
    train_date_Y1 = input_index_Y1[:input_index_Y1.index(out_sample_begin)]

    out_sample_date_Y1 = input_index_Y1[input_index_Y1.index(out_sample_begin):]

    input_index_Y2 = list(feature_list_Y2[0].index)
    tickers = ['EMB', 'GLD', 'TLT', 'IYR', 'IGIB', 'IJH']
    train_date_Y2 = input_index_Y2[:input_index_Y2.index(out_sample_begin)]

    out_sample_date_Y2 = input_index_Y2[input_index_Y2.index(out_sample_begin):]

    for i in range(len(tickers)):
        input_list_Y1 = list(feature_list_Y1[i].columns)
        input_list_Y1 .remove('Y1')

        input_Y1  = feature_list_Y1[i][input_list_Y1 ].loc[train_date_Y1 ]
        output_Y1 = np.array(feature_list_Y1[i]['Y1'].loc[train_date_Y1 ])

        x_train_Y1, x_test_Y1, y_train_Y1, y_test_Y1 = sk.model_selection.train_test_split(input_Y1, output_Y1, random_state=1,
                                                                               train_size=0.6)

        scaler_Y1 = sk.preprocessing.StandardScaler().fit(x_train_Y1)  # 通过训练集获得归一化函数模型。（也就是先减几，再除以几的函数）。在训练集和测试集上都使用这个归一化函数
        x_train_transformed_Y1 = scaler_Y1.transform(x_train_Y1)
        x_test_transformed_Y1 = scaler_Y1.transform(x_test_Y1)
        clf_Y1, result_Y1 = classifier(x_train_transformed_Y1, x_test_transformed_Y1, y_train_Y1, y_test_Y1, type)

        input_list_Y2 = list(feature_list_Y2[i].columns)
        input_list_Y2 .remove('Y2')

        input_Y2  = feature_list_Y2[i][input_list_Y2 ].loc[train_date_Y2 ]
        output_Y2 = np.array(feature_list_Y2[i]['Y2'].loc[train_date_Y2 ])

        x_train_Y2, x_test_Y2, y_train_Y2, y_test_Y2 = sk.model_selection.train_test_split(input_Y2, output_Y2, random_state=1,
                                                                               train_size=0.6)

        scaler_Y2 = sk.preprocessing.StandardScaler().fit(x_train_Y2)  # 通过训练集获得归一化函数模型。（也就是先减几，再除以几的函数）。在训练集和测试集上都使用这个归一化函数
        x_train_transformed_Y2 = scaler_Y2.transform(x_train_Y2)
        x_test_transformed_Y2 = scaler_Y2.transform(x_test_Y2)
        clf_Y2, result_Y2 = classifier(x_train_transformed_Y2, x_test_transformed_Y2, y_train_Y2, y_test_Y2, type)



        ml_model['%s' % tickers[i]] = {}
        ml_model[tickers[i]]['model1'] = clf_Y1
        ml_model[tickers[i]]['model2'] = clf_Y2
        ml_model[tickers[i]]['accuracy1'] = result_Y1
        ml_model[tickers[i]]['accuracy2'] = result_Y2
        ml_model[tickers[i]]['transform1'] = scaler_Y1
        ml_model[tickers[i]]['transform2'] = scaler_Y2
        i += 1


    view = []
    #for j in range(len(info)):
    #    y1 = ml_model[tickers[j]]['model1'].predict(info[j])
    #    y2 = ml_model[tickers[j]]['model2'].predict(info[j])
    #    y_tran = y1 * y2
    #    view += [y_tran]
    #view = np.matrix(view)
    #view = pd.DataFrame(view, index=out_sample_date, columns=tickers)


    for  i in range(len(tickers)):
        input_list_Y1 = list(feature_list_Y1[i].columns)
        input_list_Y1.remove('Y1')
        input = feature_list_Y1[i][input_list_Y1]

        input_transform_Y1 = ml_model[tickers[i]]['transform1'].transform(input.loc[out_sample_date_Y1])
        y_pre_Y1 = ml_model[tickers[i]]['model1'].predict(input_transform_Y1)
        #print(y_pre)
        input_list_Y2 = list(feature_list_Y2[i].columns)
        input_list_Y2.remove('Y2')
        input = feature_list_Y2[i][input_list_Y2]

        input_transform_Y2 = ml_model[tickers[i]]['transform2'].transform(input.loc[out_sample_date_Y2])
        y_pre_Y2 = ml_model[tickers[i]]['model2'].predict(input_transform_Y2)


        y_pre = y_pre_Y1 * y_pre_Y2
        #print(y_pre)
        print(accuracy(y_pre_Y1, y_pre_Y2, np.array(feature_list_Y1[i]['Y1'].loc[out_sample_date_Y1]), np.array(feature_list_Y2[i]['Y2'].loc[out_sample_date_Y2])))
        print(ml_model[tickers[i]]['model1'].score(input_transform_Y1,np.array(feature_list_Y1[i]['Y1'].loc[out_sample_date_Y1])))
        print(ml_model[tickers[i]]['model2'].score(input_transform_Y2,np.array(feature_list_Y2[i]['Y2'].loc[out_sample_date_Y2])))
        view += [y_pre]

    view = np.matrix(view).T

    #view = pd.DataFrame(view, index=out_sample_date, columns=tickers)
    view = pd.DataFrame(view, index=out_sample_date_Y1 , columns=tickers)
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

feature_list_Y1, feature_list_Y2 = feature_intro(close, price)

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

out_sample_begin = pd.to_datetime( ret.index[day_list.index(datetime.datetime(2016,1,14))], format="%Y-%m-%d")
type = 'Random Forest' # 'SVM' 'random forest' #  'Bayesian' 'KNN' 'Logistic'
view_final, ml_model = get_view(feature_list_Y1, feature_list_Y2, type, out_sample_begin)

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
