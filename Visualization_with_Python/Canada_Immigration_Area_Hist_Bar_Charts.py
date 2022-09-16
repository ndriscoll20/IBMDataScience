# -*- coding: utf-8 -*-
"""
Created on Sat Sep 10 11:00:34 2022

@author: Nick
"""
import numpy as np 
import pandas as pd 
import matplotlib as mpl
import matplotlib.pyplot as plt
# for name, hex in matplotlib.colors.cnames.items():
#     print(name,hex)
mpl.style.use('ggplot')

## Loading Data
canada = r'C:\Users\Nick\Documents\Python\Coursera\Visualization_with_Python\Canada.xlsx'
df_can = pd.read_excel(
    canada,
    sheet_name='Canada by Citizenship',
    skiprows=range(20),
    skipfooter=2)

## Preprocessing
df_can.drop(['AREA', 'REG', 'DEV', 'Type', 'Coverage'], axis=1, inplace=True)
df_can.rename(columns={'OdName':'Country', 'AreaName':'Continent','RegName':'Region'}, inplace=True)
df_can.columns = list(map(str, df_can.columns))
df_can.set_index('Country', inplace=True)
years = list(map(str, range(1980, 2014)))
df_can['Total'] = df_can[years].sum(axis=1)

df_can.head()
print('data dimensions: ', df_can.shape)

## Area Plots
df_can.sort_values(['Total'], ascending=False, axis=0, inplace=True)

df_top5 = df_can.head()

df_top5 = df_top5[years].transpose()
df_top5.head()

# Change the index (years) to ints for plotting
df_top5.index = df_top5.index.map(int)
df_top5.plot(kind='area',
             alpha=0.25,        #level of transparency
             stacked=False,     #Unstacked shows areas transparent
             figsize=(20, 10))  # pass a tuple (x, y) size

plt.title('Immigration Trend of Top 5 Countries')
plt.ylabel('Number of Immigrants')
plt.xlabel('Years')

plt.show()

df_can.sort_values(['Total'], axis=0, inplace=True)
df_last5 = df_can.head()
df_last5 = df_last5[years].transpose()
df_last5.index = df_last5.index.map(int)

df_last5.plot(kind='area',alpha=0.45,stacked=False,figsize=(20,10))
plt.title('5 Lowest Levels of Immigration to Canada')
plt.xlabel('Years')
plt.ylabel('Number of Immigrants')
plt.show()

## Histograms
count, bin_edges = np.histogram(df_can['2013'])
df_can['2013'].plot(kind='hist', figsize=(8, 5),xticks=bin_edges)
plt.title('Histogram of Immigration from 195 Countries in 2013')
plt.xlabel('Number of Immigrants')
plt.ylabel('Countries')

# Denmark, Sweden and Norway
df_t = df_can.loc[['Denmark','Sweden','Norway'], years].transpose()
df_t.plot(kind='hist',figsize=(8,5))

# Increase bins & make bars transparent
count, bin_edges = np.histogram(df_t, 15)
xmin = bin_edges[0] - 10   #  first bin value is 31.0, adding buffer of 10 for aesthetic purposes 
xmax = bin_edges[-1] + 10  #  last bin value is 308.0, adding buffer of 10 for aesthetic purposes
df_t.plot(kind = 'hist',
          figsize=(10,6),
          bins=15,
          xticks=bin_edges,
          stacked=True,
          xlim=(xmin,xmax),
          color=['coral','darkslateblue','mediumseagreen'])
plt.title('Histogram of Immigration from Denmark, Norway, and Sweden')
plt.ylabel('Number of Years')
plt.xlabel('Number of Immigrants')

# Greece Albania and Bulgaria
df_gab = df_can.loc[['Greece','Albania','Bulgaria'],years].transpose()
count, bin_edges = np.histogram(df_gab, 15)
df_gab.plot(kind = 'hist',
          figsize=(10,6),
          bins=15,
          xticks=bin_edges,
          alpha=0.35,
          stacked=True,
          color=['coral','darkslateblue','mediumseagreen'])
plt.title('Histogram of Immigration from Greece, Albania, and Bulgaria')
plt.ylabel('Number of Years')
plt.xlabel('Number of Immigrants')
plt.show()

## Bar Plot
# Investigate Iceland's Financial Crisis
df_ice = df_can.loc[['Iceland'],years].transpose()
df_ice.plot(kind= 'bar',figsize =(10,6), rot=90)
#Annotate Arrow:
plt.annotate('',
            xy = (32,70),
            xytext=(28,20),     #start point, index 28 (2008), population 20
            xycoords='data',
            arrowprops=dict(arrowstyle='->',connectionstyle='arc3',color='blue',lw=2))
#Annotate Text:
plt.annotate('Financial Crisis of 2008',
             xy=(28,30),
             rotation=72.5,
             va='bottom',   #vertical alignment
             ha='left')     #horizontal alignment
plt.title('Iceland Immigration to Canada')
plt.xlabel('Years')
plt.ylabel('Number of Immigrants')

#Horizontal Bar Plots
df_can.sort_values(['Total'],ascending=False,inplace=True)
df_top15 = df_can.head(15)
#df_top15 = df_top15['Total'].transpose()
#df_top15.index = df_top15.index.map(int)

df_top15.plot(kind='barh', figsize=(10,6))
plt.title('Top 15 Countries Immigrating to Canada')
plt.xlabel('Number of Immigrants')
for index, value in enumerate(df_top15):
    label = format(int(value),',')
    plt.annotate(label,xy=(value - 47000, index -0.10), color='white')
