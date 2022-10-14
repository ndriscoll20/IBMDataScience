# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 22:32:31 2022

@author: Nick
"""

import numpy as np
import pandas as pd
import pylab as pl
import scipy.optimize as opt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, jaccard_score, classification_report, log_loss
import matplotlib.pyplot as plt
import itertools


path = r'C:\Users\Nick\Documents\Python\Coursera\IBMDataScience\MachineLearningPython\ChurnData.csv'
churn_df = pd.read_csv(path)

#drop some columns: 
churn_df = churn_df[['tenure','age','address','income','ed','employ','equip','callcard','wireless','churn']]
churn_df['churn'] = churn_df['churn'].astype('int')

X = np.asarray(churn_df[['tenure','age','address','income','ed','employ','equip']])

y = np.asarray(churn_df['churn'])

X = preprocessing.StandardScaler().fit(X).transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 4)

# Solvers: ‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’, C = inverse of regularization, i.e., smaller is stronger
LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train,y_train)

y_hat = LR.predict(X_test)
y_hat_proba = LR.predict_proba(X_test)

jacc_score = jaccard_score(y_test, y_hat,pos_label=0)

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix',cmap=plt.cm.Blues):
    '''
    This function prints and plots a confusion matrix
    '''
    if normalize:
        cm = cm.astype('float')/cm.sum(axis=1)[:,np.newaxis]
        print('Normalized confusion matrix')
    else:
        print('Confusion Matrix, without normalization')
    print(cm)
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks=np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() /2.
    for i,j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), 
            horizontalalignment='center',
            color='white' if cm[i,j] > thresh else 'black')
    
    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
print(confusion_matrix(y_test, y_hat, labels=[1,0]))

cnf_matrix = confusion_matrix(y_test, y_hat, labels=[1,0])
np.set_printoptions(precision=2)

plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['churn=1','churn=0'], normalize=False, title='Confusion Matrix')

print(classification_report(y_test, y_hat))

log_loss(y_test, y_hat)
