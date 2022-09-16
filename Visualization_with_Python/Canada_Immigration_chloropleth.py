# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 22:34:35 2022

@author: Nick
"""


import pandas as pd
import numpy as np
import folium
import json

canada = r'C:\Users\Nick\Documents\Python\Coursera\Visualization_with_Python\Canada.xlsx'
countries_json = r'C:\Users\Nick\Documents\Python\Coursera\Visualization_with_Python\world_countries.json'
df_can = pd.read_excel(
    canada,
    sheet_name='Canada by Citizenship',
    skiprows=range(20),
    skipfooter=2)

world_geo = json.load(countries_json)

df_can.drop(['AREA', 'REG', 'DEV', 'Type', 'Coverage'], axis=1, inplace=True)
df_can.rename(columns={'OdName':'Country', 'AreaName':'Continent','RegName':'Region'}, inplace=True)
df_can.columns = list(map(str, df_can.columns))
df_can.set_index('Country', inplace=True)
years = list(map(str, range(1980, 2014)))
df_can['Total'] = df_can[years].sum(axis=1)

# create a numpy array of length 6 and has linear spacing from the minimum total immigration to the maximum total immigration
threshold_scale = np.linspace(df_can['Total'].min(),
                              df_can['Total'].max(),
                              6, dtype=int)
threshold_scale = threshold_scale.tolist() # change the numpy array to a list
threshold_scale[-1] = threshold_scale[-1] + 1 # make sure that the last value of the list is greater than the maximum immigration

# let Folium determine the scale.
world_map = folium.Map(location=[0, 0], zoom_start=2)
world_map.choropleth(
    geo_data=world_geo,
    data=df_can,
    columns=['Country', 'Total'],
    key_on='feature.properties.name',
    threshold_scale=threshold_scale,
    fill_color='YlOrRd', 
    fill_opacity=0.7, 
    line_opacity=0.2,
    legend_name='Immigration to Canada',
    reset=True
)
world_map