# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 14:47:02 2022

@author: Nick
"""

import numpy as np 
import pandas as pd
import scipy
import pylab
#from scipy import ndimage 
from scipy.cluster import hierarchy 
from scipy.spatial import distance_matrix 
from matplotlib import pyplot as plt 
from sklearn import manifold, datasets 
from sklearn.cluster import AgglomerativeClustering 
from sklearn.preprocessing import MinMaxScaler

path = r'C:\Users\Nick\Documents\Python\Coursera\IBMDataScience\MachineLearningPython\cars_clus.csv'
cars_df = pd.read_csv(path)
cars_df.replace('$null$',np.NaN, inplace=True)
cars_df = cars_df.dropna()
cars_df = cars_df.reset_index(drop=True)

features_df = cars_df[['engine_s','horsepow','wheelbas','width','length','curb_wgt','fuel_cap','mpg']]

x = features_df.values
min_max_scalar = MinMaxScaler()
feature_mtx = min_max_scalar.fit_transform(x)

leng = feature_mtx.shape[0]
D = scipy.zeros([leng,leng])
for i in range(leng):
    for j in range(leng):
        D[i,j] = scipy.spatial.distance.euclidean(feature_mtx[i], feature_mtx[j])
        
Z = hierarchy.linkage(D, 'complete')
max_d = 3
clusters = hierarchy.fcluster(Z, max_d, criterion='distance')

k = 5
clusters = hierarchy.fcluster(Z, k, criterion='maxclust')

fig = pylab.figure(figsize=(18,50))
def llf(id):
    return '[%s %s %s]' % (cars_df['manufact'][id], cars_df['model'][id], int(float(cars_df['type'][id])) )

dendro = hierarchy.dendrogram(Z, leaf_label_func=llf, leaf_rotation=0, leaf_font_size=12, orientation='right')


# Clustering with Scikit-Learn 
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.cm as cm 
dist_matrix = euclidean_distances(feature_mtx, feature_mtx)

Z_using_dist = hierarchy.linkage(dist_matrix, 'complete')
fig = pylab.figure(figsize=(18,50))

dendro = hierarchy.dendrogram(Z_using_dist, leaf_label_func=llf, leaf_rotation=0, leaf_font_size=12, orientation='right')

# Now Using Agglomoration
agglom = AgglomerativeClustering(n_clusters=6, linkage='complete')
agglom.fit(dist_matrix)
cars_df['cluster_'] = agglom.labels_

n_clusters = max(agglom.labels_)+1
colors = cm.rainbow(np.linspace(0, 1, n_clusters))
cluster_labels = list(range(0, n_clusters))

plt.figure(figsize=(8,7))

for color, label in zip(colors, cluster_labels):
    subset = cars_df[cars_df.cluster_ == label]
    for i in subset.index:
            plt.text(subset.horsepow[i], subset.mpg[i],str(subset['model'][i]), rotation=25) 
    plt.scatter(subset.horsepow, subset.mpg, s= subset.price*10, c=color, label='cluster'+str(label),alpha=0.5)
#    plt.scatter(subset.horsepow, subset.mpg)
plt.legend()
plt.title('Clusters')
plt.xlabel('horsepow')
plt.ylabel('mpg')

cars_df.groupby(['cluster_','type'])['cluster_'].count()
#Check Characteristics of each cluster:
agg_cars = cars_df.groupby(['cluter_','type'])['horsepow','mpg','price'].mean()

plt.figure(figsize=(16,10))
for color, label in zip(colors, cluster_labels):
    subset = agg_cars.loc[(label,),]
    for i in subset.index:
        plt.text(subset.loc[i][0]+5, subset.loc[i][2], 'type='+str(int(i)) + ', price='+str(int(subset.loc[i][3]))+'k')
    plt.scatter(subset.horsepow, subset.mpg, s=subset.price*20, c=color, label='cluster'+str(label))
plt.legend()
plt.title('Clusters')
plt.xlabel('horsepow')
plt.ylabel('mpg')