#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 18:17:56 2018
"""

'''
Backtest is the class that receive returns of portfolio as input
and output all kinds of evalution metrics like Sharpe ratio, Max drawdown
'''

#--------------------------------------------- BackTest Functions --------------------------------------------------
# Parameters Input:
    # rets_df(DataFrame): returns of the portfolios
    # mkt_ticker(string): ticker of market index
    # a(float): significance level of the VaR
# Member Functions:
    # .summary(): Calculate All Parameters & Summarize into a DataFrame  
    # .cal_ann_return(): Calculate Annualized Return(Daily)
    # .cal_ann_vol(): Calculate annualized volatility(Daily)
    # .cal_sharpe_ratio(): Sharpe Ratio
    # .cal_market_beta(): Calculate Market Beta
    # .cal_max_drawdown(): Max Drawdown
    # .cal_VaR(): Value-at-Risk
#--------------------------------------------------------------------------------------------------------------------
    
class Backtest:   
    
    def __init__(self,rets_df,mkt_ticker,a,freq):
        self.np = __import__('numpy')
        self.pd = __import__('pandas')
        self.sc = __import__('scipy.stats')
        self.rets = rets_df
        self.mkt = mkt_ticker
        self.a = a
        self.freq = freq
        print("Back Test is initialized.")
        
    def cal_ann_return(self):
        return self.rets.mean()*self.freq

    def cal_ann_vol(self):
        return self.np.sqrt(self.rets.var()*self.freq)
    
    def cal_sharpe_ratio(self):
        return self.np.sqrt(self.freq)*self.rets.mean()/self.rets.std()

    def cal_market_beta(self):
        mkt_var = self.np.var(self.rets[self.mkt])
        cov = self.pd.Series({symbol: self.rets[self.mkt].cov(self.rets[symbol]) for symbol in self.rets.columns })
        return cov/mkt_var    #self.np.ones(self.rets.shape[1])
    
    def cal_max_drawdown(self):
        net_vals = self.np.cumprod(self.rets+1)
        self.max_drawdown = []
    
        for col in list(net_vals):
            net = net_vals[col]
            maxdraw = 0
            for i in range(len(net)):
                maxlocal = (min(net[i:])-net[i])/net[i]
                if maxlocal < maxdraw:
                    maxdraw = maxlocal
            self.max_drawdown.append(maxdraw)
        
        return self.pd.Series(self.max_drawdown, index=self.rets.columns)
         
    def cal_VaR(self):
        mean = self.np.mean(self.rets)
        std = self.np.std(self.rets)
        return mean*self.freq - self.sc.stats.norm.ppf(1-self.a)*std*self.np.sqrt(self.freq)
        
    def summary(self):
        print("Running Back Test:")
        ann_ret = self.cal_ann_return()
        ann_vol = self.cal_ann_vol()
        sharpe = self.cal_sharpe_ratio()
        beta = self.cal_market_beta()
        max_drawdown = self.cal_max_drawdown()
        VaR = self.cal_VaR()
        
        params = ['Annualized Return','Annualized Volatility','Sharpe Ratio','Beta','Max Drawdown','VaR'] 
        self.summ_df = self.pd.DataFrame([ann_ret,ann_vol,sharpe,beta,max_drawdown,VaR],index=params).round(4)
        return self.summ_df
