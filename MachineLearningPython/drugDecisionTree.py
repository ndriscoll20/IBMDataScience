# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 18:34:25 2022

@author: Nick
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import sklearn.tree as tree
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt

path = r'C:\Users\Nick\Documents\Python\Coursera\IBMDataScience\MachineLearningPython\drug200.csv'
df = pd.read_csv(path, delimiter=',')

print(df.shape)

X = df[['Age','Sex','BP','Cholesterol','Na_to_K']].values

#Preprocessing
le_sex = preprocessing.LabelEncoder()
le_sex.fit(['F','M'])
X[:,1] = le_sex.transform(X[:,1])

le_BP = preprocessing.LabelEncoder()
le_BP.fit(['LOW','NORMAL','HIGH'])
X[:,2] = le_BP.transform(X[:,2])

le_Chol = preprocessing.LabelEncoder()
le_Chol.fit(['NORMAL','HIGH'])
X[:,3] = le_Chol.transform(X[:,3])

y = df['Drug']

# Split and Fit
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=3)

drug_tree = DecisionTreeClassifier(criterion='entropy',max_depth=4)
drug_tree.fit(X_train,y_train)

# Prediction
predTree = drug_tree.predict(X_test)
print(predTree[0:5])
print(y_test[0:5])

# Evaluation
print('Decision Tree Accuracy: ', metrics.accuracy_score(y_test, predTree))
tree.plot_tree(drug_tree)
plt.show()