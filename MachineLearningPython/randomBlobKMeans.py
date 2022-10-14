# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 10:35:55 2022

@author: Nick
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

#Generate radnom data: 
np.random.seed(0)

X, y = make_blobs(n_samples=5000, centers=[[4,4],[-2,-1],[2,-3],[1,1]], cluster_std=0.9)

plt.scatter(X[:,0], X[:,1],marker='.')

k_means = KMeans(n_clusters=4, init='k-means++', n_init=12)
k_means.fit(X)

#K_means labelled predictions, cluster centroids:
k_means_labels = k_means.labels_
k_means_centroids = k_means.cluster_centers_

#Plot it

fig = plt.figure(figsize = (6,4))

colors = plt.cm.Spectral(np.linspace(0,1, len(set(k_means_labels))))
ax = fig.add_subplot(1,1,1)

for k, col in zip(range(len([[4,4], [-2, -1], [2, -3], [1, 1]])), colors):
    #Create a labeled list of datapoints in or out of the cluster
    my_members = (k_means_labels == k)
    #Define the centroid
    cluster_center = k_means_centroids[k]
    #plot the datapoints
    ax.plot(X[my_members, 0], X[my_members, 1], 'w', markerfacecolor=col, marker='.')
    #Plot the centroids with a dark outline
    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,  markeredgecolor='k', markersize=6)

ax.set_title('KMeans')
# Remove ticks
ax.set_xticks(())
ax.set_yticks(())
plt.show()
