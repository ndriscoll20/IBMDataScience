# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 14:43:44 2022

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

## Pie Charts
df_continents = df_can.groupby('Continent',axis=0).sum()
print(type(df_can.groupby('Continent',axis=0)))
df_continents.head()

colors_list = ['gold','yellowgreen','lightcoral','lightskyblue','lightgreen','pink']
explode_list = [0.1,0,0,0,0.1,0.1] #offset ratio for each continent

df_continents['Total'].plot(kind='pie',
                            figsize=(14,10),
                            autopct='%1.1f%%',
                            startangle=90,
                            shadow=True,
                            labels=None,
                            pctdistance=1.12,
                            colors=colors_list,
                            explode=explode_list)

plt.title('Immigration to Canada by Continent [1980 - 2013]', y=1.12)
plt.axis('equal')
plt.legend(labels=df_continents.index, loc='upper left')

plt.show()

## Box Plots
df_japan = df_can.loc[['Japan'],years].transpose()

df_japan.plot(kind='box', figsize=(8,6))
plt.ylabel('Number of Immigrants')
plt.show()

# Using subplots
df_CI = df_can.loc[['China','India'],years].transpose()
fig=plt.figure()
ax0=fig.add_subplot(1,2,1)
ax1=fig.add_subplot(1,2,2)

df_CI.plot(kind='box', color='blue',vert=False, figsize=(20,6),ax=ax0)
ax0.set_title('Box Plots of Immigration from China and India')
ax0.set_xlabel('Number of Immigrants')
ax0.set_ylabel('Countries')

df_CI.plot(kind='line', figsize=(20,6),ax=ax1)
ax1.set_title('Line Plots of Immigration from China and India')
ax1.set_xlabel('Number of Immigrants')
ax1.set_ylabel('Countries')

plt.show()

# Aggregate data by decade: 
df_can.sort_values(['Total'],ascending=False, axis=0, inplace=True)
df_top15 = df_can.head(15)
eighties = list(map(str,range(1980,1990)))
ninties = list(map(str,range(1990,2000)))
naughts = list(map(str,range(2000,2014)))

sum80s = df_top15.loc[:,eighties].sum(axis=1)
sum90s = df_top15.loc[:,ninties].sum(axis=1)
sum00s = df_top15.loc[:,naughts].sum(axis=1)

new_df = pd.DataFrame({'1980s': sum80s, '1990s': sum90s, '2000s':sum00s})
new_df.head()

## Scatter Plotting
# Data prep for yearly totals: 
df_total = pd.DataFrame(df_can[years].sum(axis=0))
df_total.index = map(int, df_total.index)
df_total.reset_index(inplace=True)
df_total.columns = ['year','total']
df_total.head()

df_total.plot(kind='scatter', x='year',y='total', figsize=(10,6), color='darkblue')
plt.title('Total Immigration to Candada [1980-2013]')
plt.xlabel('Year')
plt.ylabel('Total Immigrants')
plt.show()

# Fit a regression line
x = df_total['year']
y = df_total['total']
fit = np.polyfit(x,y,deg=1)

df_total.plot(kind='scatter',x='year',y='total', figsize=(10,6), color='darkblue')
plt.title('Total Immigration to Candada [1980-2013]')
plt.xlabel('Year')
plt.ylabel('Total Immigrants')
plt.plot(x, fit[0] * x + fit[1], color='red')
plt.annotate('y={0:.0f} x + {1:.0f}'.format(fit[0], fit[1]), xy=(2000,15000))

#'No. Immigrants = {0:.0f} * Year + {1:.0f}'.format(fit[0], fit[1]) 

years = list(map(str,range(1980,2014)))
df_countries = df_can.loc[['Denmark','Norway','Sweden'],years].transpose()

df_total = pd.DataFrame(df_countries.sum(axis=1))
df_total.reset_index(inplace=True)
df_total.columns = ['year','total']
df_total['year'] = df_total['year'].astype(int)
df_total.head()

## Bubble Plots
df_can_t = df_can[years].transpose()
df_can_t.index = map(int, df_can_t.index)
df_can_t.index.name = 'Year'
df_can_t.reset_index(inplace=True)

norm_brazil = (df_can_t['Brazil'] - df_can_t['Brazil'].min()) / (df_can_t['Brazil'].max() - df_can_t['Brazil'].min())
norm_argentina = (df_can_t['Argentina'] - df_can_t['Argentina'].min()) / (df_can_t['Argentina'].max() - df_can_t['Argentina'].min())

ax0 = df_can_t.plot(kind='scatter',x='Year',y='Brazil',figsize=(14,8),
                    alpha=0.5, color='green', s=norm_brazil *2000+10,
                    xlim=(1975, 2015))
ax1 = df_can_t.plot(kind='scatter',x='Year',y='Argentina',
                    alpha=0.5, color='blue', s=norm_argentina *2000+10,
                    ax=ax0)
ax0.set_ylabel('Number of Immigrants')
ax0.set_title('Immigration from Brazil and Argentina from 1980 to 2013')
ax0.legend(['Brazil','Argentina'],loc = 'upper left', fontsize = 'x-large')



norm_china = (df_can_t['China'] - df_can_t['China'].min()) / (df_can_t['China'].max() - df_can_t['China'].min())

norm_india = (df_can_t['India'] - df_can_t['India'].min()) / (df_can_t['India'].max() - df_can_t['India'].min())

#China
ax0 = df_can_t.plot(kind='scatter', x='Year', y='China', figsize=(14, 8),
                    alpha=0.5, color='green',s=norm_china * 2000 + 10, xlim=(1975, 2015))

# India
ax1 = df_can_t.plot(kind='scatter', x='Year', y='India', alpha=0.5,
                    color="blue", s=norm_india * 2000 + 10, ax=ax0)

ax0.set_ylabel('Number of Immigrants')
ax0.set_title('Immigration from China and India from 1980 to 2013')
ax0.legend(['India', 'China'], loc='upper left', fontsize='x-large')