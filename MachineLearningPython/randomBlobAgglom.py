# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 14:31:17 2022

@author: Nick
"""

import numpy as np 
import pandas as pd
from scipy import ndimage 
from scipy.cluster import hierarchy 
from scipy.spatial import distance_matrix 
from matplotlib import pyplot as plt 
from sklearn import manifold, datasets 
from sklearn.cluster import AgglomerativeClustering 
from sklearn.datasets import make_blobs 

X1, y1 = make_blobs(n_samples=50, centers=[[1,1],[2,5],[4,3]], cluster_std=0.9)

plt.scatter(X1[:,0],X1[:,1], marker='.')

agglom = AgglomerativeClustering(n_clusters=4, linkage='complete')
agglom.fit(X1, y1)


#plt.figure(figsize=(6,4))

# Scale the points down
# x_min, x_max = np.min(X1, axis=0), np.max(X1, axis=0)
# X1 = (X1 - x_min) / (x_max - x_min)

for i in range(X1.shape[0]):
    plt.text(X1[i, 0], X1[i, 1], str(y1[i]),
             color=plt.cm.nipy_spectral(agglom.labels_[i] / 10.),
             fontdict={'weight': 'bold', 'size': 9})
    
# Remove the x ticks, y ticks, x and y axis
plt.xticks([])
plt.yticks([])
plt.axis('off')
#plt.scatter(X1[:, 0], X1[:, 1], marker='.')
plt.show()

dist_matrix = distance_matrix(X1, X1)
Z = hierarchy.linkage(dist_matrix, 'average')

dendro = hierarchy.dendrogram(Z)