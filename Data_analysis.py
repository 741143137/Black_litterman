#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 19:33:03 2019

Visualize and test data
"""

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
tickers = ['SPY']

st = datetime.datetime(2007,1,1)
end = datetime.datetime(2015,10,31)
#out sample
#st = datetime.datetime(2014,10,31)
#end = datetime.datetime(2018,10,31)

#close = pd.DataFrame(columns = tickers)
close = pdr.get_data_yahoo(tickers,start = st,end = end)["Adj Close"]
ret = close.pct_change().dropna()
