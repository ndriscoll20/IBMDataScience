# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 10:35:46 2022

@author: 1109336
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier


def plot_confusion_matrix(y,y_predict):
    #Confusion Matrix Plotter Function
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y, y_predict)
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, ax = ax); #annot=True to annotate cells
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix'); 
    ax.xaxis.set_ticklabels(['did not land', 'land']); ax.yaxis.set_ticklabels(['did not land', 'landed'])
    
path = r'C:\Users\1109336\Documents\Python\Coursera\DataScienceCapstone'
data = pd.read_csv(path+'\dataset_part_2.csv')
X = pd.read_csv(path+'\dataset_part_3.csv')

Y = data['Class'].to_numpy()
transform = preprocessing.StandardScaler()
X = transform.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=2)

# Logistic Regression GridSearchCV
parameters = {'C':[0.01,0.1,1], 'penalty':['l2'], 'solver':['lbfgs']}
lr = LogisticRegression()
#X_lr = lr.fit(X_train, Y_train)
logreg_cv = GridSearchCV(lr, param_grid=parameters, cv=10)
logreg_cv.fit(X_train, Y_train)
#logreg_cv

print('(LR) Tuned hyperparameters (best parameters): ', logreg_cv.best_params_)
print('(LR) Accuracy :', logreg_cv.best_score_)
print('(LR) Test Data Acuracy: ', logreg_cv.score(X_test, Y_test))

yhat = logreg_cv.predict(X_test)
plot_confusion_matrix(Y_test, yhat)

# SVM GridSearchCV
parameters_svm = {'kernal':('linear','rbf','poly','sigmoid'),
              'C': np.logspace(-3, 3, 5),
              'gamma': np.logspace(-3, 3, 5)}
svm = SVC()
svm_cv = GridSearchCV(svm, parameters_svm, cv=10)
svm_cv.fit(X_train, Y_train)

print('(SVM) Tuned hyperparameters (best parameters): ', svm_cv.best_params_)
print('(SVM) Accuracy :', svm_cv.best_score_)
print('(SVM) Test Data Acuracy: ', svm_cv.score(X_test, Y_test))

yhat_svm = svm_cv.predict(X_test)
plot_confusion_matrix(Y_test, yhat_svm)

# Decision Tree GridSearchCV
parameters_tree = {'criterion': ['gini', 'entropy'],
     'splitter': ['best', 'random'],
     'max_depth': [2*n for n in range(1,10)],
     'max_features': ['auto', 'sqrt'],
     'min_samples_leaf': [1, 2, 4],
     'min_samples_split': [2, 5, 10]}

tree = DecisionTreeClassifier()
tree_cv = GridSearchCV(tree, parameters_tree, cv=10)
tree_cv.fit(X_train, Y_train)

print('(Tree) Tuned hyperparameters (best parameters): ', tree_cv.best_params_)
print('(Tree) Accuracy :', tree_cv.best_score_)
print('(Tree) Test Data Acuracy: ', tree_cv.score(X_test, Y_test))

yhat_tree = tree_cv.predict(X_test)
plot_confusion_matrix(Y_test, yhat_tree)

#KNN
parameters_knn = {'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
              'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
              'p': [1,2]}
KNN = KNeighborsClassifier()
knn_cv = GridSearchCV(KNN, parameters_knn, cv=10)
knn_cv.fit(X_train, Y_train)

print('(KNN) Tuned hyperparameters (best parameters): ', knn_cv.best_params_)
print('(KNN) Accuracy :', knn_cv.best_score_)
print('(KNN) Test Data Acuracy: ', knn_cv.score(X_test, Y_test))

yhat_knn = knn_cv.predict(X_test)
plot_confusion_matrix(Y_test, yhat_knn)

