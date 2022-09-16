# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 16:31:44 2022

@author: Nick
"""

import yfinance as yf
import pandas as pd
apple = yf.Ticker("AAPL")

apple_info=apple.info
apple_info

apple_info['country']
apple_share_price_data = apple.history(period="max")
apple_share_price_data.head()
apple_share_price_data.reset_index(inplace=True)
apple_share_price_data.plot(x="Date", y="Open")
apple.dividends.plot()

amd = yf.Ticker('AMD')
amd.info['country']
amd.info['sector']

#Question: plot the max history of Volume
stock_data = amd.history(period='max')
stock_data['Volume'][0]
amd_history = amd.history(period='max')
amd_history.reset_index(inplace=True)
amd_history.plot(x='Date',y='Volume')
#What was the volume on opening day?
amd_history.iloc[0,:]