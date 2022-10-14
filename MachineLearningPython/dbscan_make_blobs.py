# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 21:57:08 2022

@author: Nick
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

# Generate data 
def createDataPoints(centroidLocation, numSamples, clusterDeviation):
    '''
    Enhance make_blobs function by standardizign the features
    '''
    X, y = make_blobs(n_samples=numSamples, centers=centroidLocation,
                      cluster_std=clusterDeviation)
    X = StandardScaler().fit_transform(X)
    return X, y

X, y = createDataPoints([[4,3],[2,-1],[-1,4]], 1500, 0.5)

epsilon = 0.3
minSamples = 7
db = DBSCAN(eps=epsilon, min_samples=minSamples).fit(X)
labels = db.labels_
# print(labels)

core_samples_mask = np.zeros_like(db.labels_,dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
# print(core_samples_mask)
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
unique_labels = set(labels)
# print(n_clusters, ' ', unique_labels)

colors = plt.cm.Spectral(np.linspace(0,1,len(unique_labels)))

for k, col in zip(unique_labels, colors):
    if k == -1: #Outlier
        col = 'k'
    class_member_mask = (labels==k)
    
    xy = X[class_member_mask & core_samples_mask]
    plt.scatter(xy[:,0], xy[:,1], s=50, c=[col], marker=u'o', alpha=0.5)
    
    xy = X[class_member_mask & ~core_samples_mask]
    plt.scatter(xy[:,0], xy[:,1], s=50, c=[col], marker=u'o', alpha=0.5)
plt.title('DBSCAN')
plt.show()

#Compare same data w/ k-means
from sklearn.cluster import KMeans 
k = 3
k_means3 = KMeans(init = "k-means++", n_clusters = k, n_init = 12)
k_means3.fit(X)
fig = plt.figure(figsize=(6, 4))
ax = fig.add_subplot(1, 1, 1)
for k, col in zip(range(k), colors):
    my_members = (k_means3.labels_ == k)
    plt.scatter(X[my_members, 0], X[my_members, 1],  c=col, marker=u'o', alpha=0.5)
plt.title('K-means')
plt.show()