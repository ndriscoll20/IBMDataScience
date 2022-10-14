# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 18:00:21 2022

@author: Nick
"""

import numpy as np
import pandas as pd
import pylab as pl
import scipy.optimize as opt
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, jaccard_score
import matplotlib.pyplot as plt
import itertools
from plot_confusion_matrix import *

path = r'C:\Users\Nick\Documents\Python\Coursera\IBMDataScience\MachineLearningPython\cell_samples.csv'
cell_df = pd.read_csv(path)

ax = cell_df[cell_df['Class'] == 4][0:50].plot(kind='scatter',
                                               x='Clump', y='UnifSize',
                                               color='DarkBlue', label='malignant');
cell_df[cell_df['Class'] == 2][0:50].plot(kind='scatter', 
                                          x='Clump', y='UnifSize', color='Yellow', 
                                          label='benign', ax=ax);
plt.show()

# Pre-processing
#print(cell_df.dtypes)
cell_df = cell_df[pd.to_numeric(cell_df['BareNuc'], errors='coerce').notnull()]
cell_df['BareNuc'] = cell_df['BareNuc'].astype('int')
#print(cell_df.dtypes)

feature_df = cell_df[['Clump','UnifSize','UnifShape','MargAdh','SingEpiSize',
                      'BareNuc','BlandChrom','NormNucl','Mit']]
X = np.asarray(feature_df)

cell_df['Class'] = cell_df['Class'].astype('int')
y = np.asarray(cell_df['Class'])

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state=4)

# RBF Kernel
clf = svm.SVC(kernel='rbf')
clf.fit(X_train, y_train)

y_hat = clf.predict(X_test)

cnf_matrix = confusion_matrix(y_test, y_hat, labels = [2,4])

np.set_printoptions(precision=2)

print(classification_report(y_test, y_hat))

plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['Benign(2)','Malignant(4)'],
                      normalize=False, title='Confusion Matrix')

print(f1_score(y_test, y_hat, average='weighted'))
print(jaccard_score(y_test, y_hat, pos_label=2))

# Linear Kernel
clf2 = svm.SVC(kernel='linear')
clf2.fit(X_train, y_train)
y_hat2 = clf2.predict(X_test)
print(f1_score(y_test,y_hat2,average='weighted'))
print(jaccard_score(y_test,y_hat2,pos_label=2))