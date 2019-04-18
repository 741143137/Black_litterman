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
y2 = abs((ret -ret.shift(-1))/rolling_sd)
y1 = np.zeros([N,M])
y1[ret<0] = -1
y1[ret>0] = 1


    