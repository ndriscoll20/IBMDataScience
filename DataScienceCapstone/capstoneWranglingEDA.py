# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 14:13:43 2022

@author: 1109336
"""
import pandas as pd
import numpy as np

path = r'C:\Users\1109336\Documents\Python\Coursera\DataScienceCapstone\dataset_part_1.csv'
df = pd.read_csv(path)

print(df.isnull().sum()/df.count()*100)
print(df.dtypes())

df['LaunchSite'].value_counts()
df['Orbit'].value_counts()
landing_outcomes = df['Outcome'].value_counts()

for i, outcome in enumerate(landing_outcomes.keys()):
    print(i, outcome)
    
bad_outcomes=set(landing_outcomes.keys()[[1,3,5,6,7]]) #False or none landings

landing_class = np.zeros(df['Outcome'].shape[0])
for i in range(1, len(landing_class)):
    if df['Outcome'][i] in bad_outcomes:
        landing_class[i] = 0
    else:
        landing_class[i] = 1
        
df['Class'] = landing_class
print(df[['Class']].head())
print(df.head())

df.to_csv('dataset_part_2.csv',index=False)