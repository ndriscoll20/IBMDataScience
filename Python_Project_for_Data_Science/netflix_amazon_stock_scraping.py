# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 16:30:24 2022

@author: Nick
"""

import pandas as pd
import requests
from bs4 import BeautifulSoup

#Netflix data scraping
url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-PY0220EN-SkillsNetwork/labs/project/netflix_data_webpage.html"

data  = requests.get(url).text

soup = BeautifulSoup(data, 'html5lib')

netflix_data = pd.DataFrame(columns=["Date", "Open", "High", "Low", "Close", "Volume"])

# First we isolate the body of the table which contains all the information
# Then we loop through each row and find all the column values for each row
for row in soup.find("tbody").find_all('tr'):
    col = row.find_all("td")
    date = col[0].text
    Open = col[1].text
    high = col[2].text
    low = col[3].text
    close = col[4].text
    adj_close = col[5].text
    volume = col[6].text
    
    # Finally we append the data of each row to the table
    netflix_data = netflix_data.append({"Date":date, "Open":Open, "High":high, "Low":low, "Close":close, "Adj Close":adj_close, "Volume":volume}, ignore_index=True)    
    
netflix_data.head()

read_html_pandas_data = pd.read_html(url)

read_html_pandas_data = pd.read_html(str(soup))
netflix_dataframe = read_html_pandas_data[0]

netflix_dataframe.head()

#Amazon data scraping
url2 = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-PY0220EN-SkillsNetwork/labs/project/amazon_data_webpage.html'
data2 = requests.get(url2).text
soup2 = BeautifulSoup(data2, 'html5lib')
soup2.title
soup2.find('tr').prettify()

amazon_data = pd.DataFrame(columns=["Date", "Open", "High", "Low", "Volume", "Adj Close*","Close*"])

for row in soup2.find("tbody").find_all("tr"):
    col       = row.find_all("td")
    date      = col[0].text
    Open      = col[1].text
    high      = col[2].text
    low       = col[3].text
    close     = col[4].text
    adj_close = col[5].text
    volume    = col[6].text
    
    amazon_data = amazon_data.append({"Date":date, "Open":Open, "High":high, "Low":low, "Volume":volume, "Adj Close*":adj_close,"Close*":close}, ignore_index=True)
   
#Question: what is the Open of the last row of amazon
amazon_data['Open'][(len(amazon_data['Open'])-1)]
