# -*- coding: utf-8 -*-
"""
Created on Sat Sep 24 10:29:28 2022

@author: Nick
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing, metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as knn

path = r'C:\Users\Nick\Documents\Python\Coursera\IBMDataScience\MachineLearningPython\teleCust1000t.csv'
#path = r'C:\Users\Nick\OneDrive\Documents\Python\Coursera\IBMDataScience\MachineLearningPython'
df = pd.read_csv(path)
#Explore
df.columns
df['custcat'].value_counts()
#df.hist(column='income',bins=50)

#Convert to Numpy array to use w/ scikit-learn
X = df[['region','tenure','age','marital','address','income','ed','employ','retire','gender','reside']].values

y = df['custcat'].values

#Normalize
X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
print('Train set: ', X_train.shape, y_train.shape)
print('Test set: ', X_test.shape, y_test.shape)

k= 6
neigh = knn(n_neighbors = k).fit(X_train, y_train)

#Prediction
y_hat = neigh.predict(X_test)

#Metrics
print('Training Set Accuracy: ', metrics.accuracy_score(y_train, neigh.predict(X_train)))
print('Test Set Accuracy: ', metrics.accuracy_score(y_test, y_hat))

Ks = 10

mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))

for ii in range(1,Ks):
    
    neigh = knn(n_neighbors = ii).fit(X_train, y_train)
    y_hat = neigh.predict(X_test)
    
    mean_acc[ii-1] = metrics.accuracy_score(y_test, y_hat)
    std_acc[ii-1] = np.std(y_hat==y_test)/np.sqrt(y_hat.shape[0])


plt.plot(range(1,Ks), mean_acc, 'g')
plt.fill_between(range(1,Ks), mean_acc-1*std_acc, mean_acc+1*std_acc,alpha =0.10)
plt.fill_between(range(1,Ks), mean_acc-3*std_acc, mean_acc+3*std_acc,alpha =0.10, color = 'green')

plt.legend(('Accuracy','+/-1 std','+/-3 std'))
plt.ylabel('Accuracy')
plt.xlabel('Number of Neighbors (k)')
plt.tight_layout()
plt.show()
print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1) 
