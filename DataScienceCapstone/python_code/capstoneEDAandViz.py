# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 08:52:45 2022

@author: Nick
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('dataset_part_2.csv')

#Plot Flightnumber vs Payload Mass categorized by Success
sns.catplot(x='FlightNumber', y='PayloadMass', hue='Class', data=df, aspect=5)
plt.xlabel('Flight Number', fontsize=20)
plt.ylabel('Payload Mass (kg)', fontsize=20)

sns.catplot(x="FlightNumber", y="LaunchSite", hue="Class", data=df, aspect = 5)
plt.xlabel("Flight Number",fontsize=20)
plt.ylabel("Launch Site",fontsize=20)
plt.show()

sns.catplot(x="PayloadMass", y="LaunchSite", hue="Class", data=df, aspect = 5)
plt.xlabel("Playload Mass (kg)",fontsize=20)
plt.ylabel("Launch Site",fontsize=20)
plt.show()


df.groupby(['Orbit']).mean()['Class'].sort_values().plot(kind='bar')
plt.show()

sns.catplot(x='FlightNumber',y='Orbit', hue='Class', data=df)
plt.xlabel('FlightNumber')
plt.ylabel('Orbit Type')
plt.show()

sns.catplot(x='PayloadMass',y='Orbit', hue='Class', data=df)
plt.xlabel('Payload Mass (kg)')
plt.ylabel('Orbit Type')
plt.show()

year=[]
def Extract_year(date):
    for i in df["Date"]:
        year.append(i.split("-")[0])
    return year

Extract_year(df['Date'])
df['Year'] = year

df.groupby(['Year']).sum()['Class'].sort_values().plot(kind='line')

features = df[['FlightNumber', 'PayloadMass', 'Orbit', 'LaunchSite', 'Flights', 'GridFins', 'Reused', 'Legs', 'LandingPad', 'Block', 'ReusedCount', 'Serial']]

features_one_hot = pd.get_dummies(features, columns=['Orbit','LaunchSite','LandingPad','Serial'])
features_one_hot = features_one_hot.astype('float64')
features_one_hot.to_csv('dataset_part_3.csv',index = False)
