# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 13:26:47 2022

@author: 1109336
"""
import pandas as pd
import requests
from bs4 import BeautifulSoup

url = 'https://finance.yahoo.com/quote/NFLX/history?p=NFLX'

data = requests.get(url).text

soup = BeautifulSoup(data, 'html5lib')

netflix_data = pd.DataFrame(columns = ['Date','Open','High','Low','Close','Adj Close','Volume'])

for row in soup.find('tbody').find_all('tr'):
    col = row.find_all('td')
    date      = col[0].text
    Open      = col[1].text
    high      = col[2].text
    low       = col[3].text
    close     = col[4].text
    adj_close = col[5].text
    volume    = col[6].text
    
    netflix_data = netflix_data.append({'Date':date,'Open':Open,'High':high,'Low':low,'Close':close,'Adj Close':adj_close,'Volume':volume}, ignore_index=True)
    
netflix_data.head()

read_html_pandas_data = pd.read_html(url)
read_html_pandas_data = pd.read_html(str(soup))

