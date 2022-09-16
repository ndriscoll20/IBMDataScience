# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 16:05:13 2022

@author: 1109336
"""

import yfinance as yf
import pandas as pd
import requests
from bs4 import BeautifulSoup
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Inputs: stock_data dataframe (with Date and Close), revenue_date dataframe (with Date and Revenue), and the name of the stock)
def make_graph(stock_data, revenue_data, stock):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("Historical Share Price", "Historical Revenue"), vertical_spacing = .3)
    stock_data_specific = stock_data[stock_data.Date <= '2021--06-14']
    revenue_data_specific = revenue_data[revenue_data.Date <= '2021-04-30']
    fig.add_trace(go.Scatter(x=pd.to_datetime(stock_data_specific.Date, infer_datetime_format=True), y=stock_data_specific.Close.astype("float"), name="Share Price"), row=1, col=1)
    fig.add_trace(go.Scatter(x=pd.to_datetime(revenue_data_specific.Date, infer_datetime_format=True), y=revenue_data_specific.Revenue.astype("float"), name="Revenue"), row=2, col=1)
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Price ($US)", row=1, col=1)
    fig.update_yaxes(title_text="Revenue ($US Millions)", row=2, col=1)
    fig.update_layout(showlegend=False,
    height=900,
    title=stock,
    xaxis_rangeslider_visible=True)
    fig.show()
    
#Using yfinance
tesla = yf.Ticker('TSLA')
tesla_data = tesla.history(period='max')
tesla_data.reset_index(inplace=True)
tesla_data.head()

#Using Webscraping
url = 'https://www.macrotrends.net/stocks/charts/TSLA/tesla/revenue?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDeveloperSkillsNetworkPY0220ENSkillsNetwork23455606-2022-01-01'
r = requests.get(url)
html_data = r.text
#html_data
soup = BeautifulSoup(html_data,'html5lib')
soup.find_all('td')

#Beautiful Soup Approach
tesla_tables = soup.find_all('table')
for index, table in enumerate(tesla_tables):
    if('Tesla Quarterly Revenue' in str(table)):
        tesla_table_index = index
        
tesla_revenue_soup = pd.DataFrame(columns= ['Date','Revenue']
                                  
for row in tesla_tables[tesla_table_index].tbody.find_all('tr'):
    col = row.find_all('td')
    if(col!= []):
        date = col[0].text
        revenue = col[1].text.replace("$",'').replace(',','')
        tesla.revenue = tesla_revenue.append({'Date':date,'Revenue':revenue}, ignore_index=True)

# Pandas Read HTML Approach
cols = ['Date','Revenue']
tesla_revenue = pd.read_html(url, match='Tesla Quarterly Revenue', flavor='bs4')[0]
tesla_revenue.columns = cols
#Remove Comma and Dollar Sign from revenue column
tesla_revenue['Revenue'] = tesla_revenue['Revenue'].str.replace(',|\$',"")
#Remove null & empty strings
tesla_revenue.dropna(inplace=True)
tesla_revenue=tesla_revenue[tesla_revenue['Revenue'] != '']

#Display the last 5 rows 
tesla_revenue.tail()
make_graph(tesla_data, tesla_revenue,'Tesla')

#Do the same for GME
url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-PY0220EN-SkillsNetwork/labs/project/stock.html'
r = requests.get(url)
html_data = r.text
soup = BeautifulSoup(html_data, 'html5lib')
cols = ['Date','Revenue']
gme_revenue = pd.read_html(url, match= 'GameStop Quarterly Revenue',flavor='bs4')[0]
gme_revenue.columns = cols
gme_revenue['Revenue'] = gme_revenue['Revenue'].str.replace(',|\$','')
gme_revenue.dropna(inplace=True)
gme_revenue=gme_revenue[gme_revenue['Revenue'] != '']
gme_revenue.tail()

make_graph(gme_data, gme_revenue,'GameStop')
