# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 22:39:13 2022

@author: Nick
"""

import numpy as np
import pandas as pd
from mpl_toolkits.basemap import Basemap

import matplotlib.pyplot as plt
from pylab import rcParams
from sklearn.cluster import DBSCAN
import sklearn.utils
from sklearn.preprocessing import StandardScaler

path = r'C:\Users\Nick\Documents\Python\Coursera\IBMDataScience\MachineLearningPython\weather-stations.csv'
wdf = pd.read_csv(path)

wdf = wdf[pd.notnull(wdf['Tm'])]  #mean temperature
wdf = wdf.reset_index(drop=True)

# Map Parameters
#rcParams['figure.figsize'] = (14,10)

llon=-140; ulon=-50; llat=40; ulat=65

wdf = wdf[(wdf['Long'] > llon) & (wdf['Long'] < ulon) & (wdf['Lat'] > llat) &(wdf['Lat'] < ulat)]

my_map = Basemap(projection='merc',
            resolution = 'l', area_thresh = 1000.0,
            llcrnrlon=llon, llcrnrlat=llat, #min longitude (llcrnrlon) and latitude (llcrnrlat)
            urcrnrlon=ulon, urcrnrlat=ulat) #max longitude (urcrnrlon) and latitude (urcrnrlat)

my_map.drawcoastlines()
my_map.drawcountries()
# my_map.drawmapboundary()
my_map.fillcontinents(color = 'white', alpha = 0.3)
my_map.shadedrelief()

xs,ys = my_map(np.asarray(wdf.Long), np.asarray(wdf.Lat))
wdf['xm']= xs.tolist()
wdf['ym'] =ys.tolist()

for index,row in wdf.iterrows():
#   x,y = my_map(row.Long, row.Lat)
   my_map.plot(row.xm, row.ym,markerfacecolor =([1,0,0]),  marker='o', markersize= 5, alpha = 0.75)
#plt.text(x,y,stn)
plt.show()

sklearn.utils.check_random_state(1000)

##Cluster based on Location (Lat,Lon)
Clus_data_set = wdf[['xm','ym']]
Clus_data_set = np.nan_to_num(Clus_data_set)
Clus_data_set = StandardScaler().fit_transform(Clus_data_set)

#Compute DBSCAN
db = DBSCAN(eps=0.15, min_samples=10).fit(Clus_data_set)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_]=True
labels = db.labels_
wdf['Clus_db'] = labels

realClusterNum = len(set(labels)) - (1 if -1 in labels else 0)
clusterNum= len(set(labels))

set(labels)

#Visualize the Clusters on the map
rcParams['figure.figsize'] = (14,10)
my_map = Basemap(projection='merc', 
                 area_thresh = 1000.0,
                 llcrnrlon = llon, llcrnrlat = llat, 
                 urcrnrlon = ulon, urcrnrlat = ulat)

my_map.drawcoastlines()
my_map.drawcountries()
my_map.fillcontinents(color='white',alpha=0.3)
my_map.shadedrelief()

colors = plt.get_cmap('jet')(np.linspace(0.0, 0.1, clusterNum))

for clust_number in set(labels):
    c=(([0.4, 0.4,0.4]) if clust_number == -1 else colors[np.int(clust_number)])
    clust_set = wdf[wdf.Clus_db == clust_number]
    my_map.scatter(clust_set.xm, clust_set.ym, color=c, marker='o', s=20, alpha=0.85)
    if clust_number != -1:
        cenx = np.mean(clust_set.xm)
        ceny = np.mean(clust_set.ym)
        plt.text(cenx, ceny, str(clust_number), fontsize=25, color='red')
        print('Cluster '+str(clust_number)+', Avg Temp: '+str(np.mean(clust_set.Tm)))
        
##Cluster based on Location, mean, max and min temperature
Clus_data_set = wdf[['xm','ym','Tx','Tm','Tn']]
Clus_data_set = np.nan_to_num(Clus_data_set)
Clus_data_set = StandardScaler().fit_transform(Clus_data_set)

#Compute DBSCAN
db = DBSCAN(eps=0.3, min_samples=10).fit(Clus_data_set)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_
wdf['Clus_Db'] = labels

realClusterNum=len(set(labels)) - (1 if -1 in labels else 0)
clusterNum = len(set(labels))

print(wdf[['Stn_Name','Tx','Tm','Clus_Db']].head(5))

#Visualize Clusters based on location and Temp
rcParams['figure.figsize'] = (14,10)
my_map = Basemap(projection='merc', 
                 area_thresh = 1000.0,
                 llcrnrlon = llon, llcrnrlat = llat, 
                 urcrnrlon = ulon, urcrnrlat = ulat)

my_map.drawcoastlines()
my_map.drawcountries()
my_map.fillcontinents(color='white',alpha=0.3)
my_map.shadedrelief()

colors = plt.get_cmap('jet')(np.linspace(0.0, 0.1, clusterNum))

for clust_number in set(labels):
    c=(([0.4, 0.4,0.4]) if clust_number == -1 else colors[np.int(clust_number)])
    clust_set = wdf[wdf.Clus_db == clust_number]
    my_map.scatter(clust_set.xm, clust_set.ym, color=c, marker='o', s=20, alpha=0.85)
    if clust_number != -1:
        cenx = np.mean(clust_set.xm)
        ceny = np.mean(clust_set.ym)
        plt.text(cenx, ceny, str(clust_number), fontsize=25, color='red')
        print('Cluster '+str(clust_number)+', Avg Temp: '+str(np.mean(clust_set.Tm)))