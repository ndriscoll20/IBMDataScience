# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 15:59:26 2022

@author: Nick
"""

import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import matplotlib.ticker as ticker
import seaborn as sns
from sklearn import preprocessing, metrics, svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import jaccard_score, f1_score, log_loss, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier as knn
import sklearn.tree as tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

df_loan = pd.read_csv('loan_train.csv')

df_loan['due_date'] = pd.to_datetime(df_loan['due_date'])
df_loan['effective_date'] = pd.to_datetime(df_loan['effective_date'])
df_loan['education'].replace('Bechalor', 'Bachelor',inplace=True) #Fix spelling issue


#Some EDA
bins = np.linspace(df_loan.Principal.min(), df_loan.Principal.max(), 10)
g = sns.FacetGrid(df_loan, col='Gender', hue='loan_status', palette='Set1',col_wrap=2)
g.map(plt.hist, 'Principal', bins=bins, ec='k')
g.fig.subplots_adjust(top=0.8)
g.fig.suptitle('Loan Status by Principal')
g.axes[-1].legend()
plt.show()

bins = np.linspace(df_loan.age.min(), df_loan.age.max(), 10)
g = sns.FacetGrid(df_loan, col='Gender', hue='loan_status', palette='Set1',col_wrap=2)
g.map(plt.hist, 'age',bins = bins, ec ='k')
g.fig.subplots_adjust(top=0.8)
g.fig.suptitle('Loan Status by Age')
g.axes[-1].legend()
plt.show()

df_loan['dayofweek'] = df_loan['effective_date'].dt.dayofweek
bins = np.linspace(df_loan.dayofweek.min(), df_loan.dayofweek.max(), 7)
g = sns.FacetGrid(df_loan, col='Gender',hue='loan_status',palette='Set1',col_wrap=2)
g.map(plt.hist, 'dayofweek',bins=bins, ec='k')
g.fig.subplots_adjust(top=0.8)
g.fig.suptitle('Loan Status by Day of Week Origination')
g.axes[-1].legend()
plt.show()

df_loan['weekend']= df_loan['dayofweek'].apply(lambda x: 1 if (x>3) else 0)

df_loan.groupby(['Gender'])['loan_status'].value_counts(normalize=True)
df_loan['Gender'].replace(to_replace=['male','female'], value=[0,1], inplace=True)

df_loan.groupby(['education'])['loan_status'].value_counts(normalize=True)
df_loan['education'].value_counts()
Feature = df_loan[['Principal','terms','age','Gender','weekend']]
Feature = pd.concat([Feature, pd.get_dummies(df_loan['education'])],axis=1)
Feature.drop(['Master or Above'], axis = 1, inplace=True) #only 2 instances

X = Feature
y = df_loan['loan_status'].values

X=preprocessing.StandardScaler().fit(X).transform(X)

# Classification Algorithms
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=4)


## KNN

Ks = 15

mean_acc = np.zeros(Ks-1)
std_acc = np.zeros(Ks-1)

for k in range(1,Ks):
    neigh = knn(n_neighbors=k).fit(X_train, y_train)
    y_hat = neigh.predict(X_test)
    
    mean_acc[k-1] = metrics.accuracy_score(y_test, y_hat)
    std_acc[k-1] = np.std(y_hat==y_test)/np.sqrt(y_hat.shape[0])
    
plt.plot(range(1,Ks), mean_acc, 'g')
plt.fill_between(range(1,Ks), mean_acc-1*std_acc, mean_acc+1*std_acc,alpha =0.10)
plt.fill_between(range(1,Ks), mean_acc-3*std_acc, mean_acc+3*std_acc,alpha =0.10, color = 'green')

plt.legend(('Accuracy','+/-1 std','+/-3 std'))
plt.ylabel('Accuracy')
plt.xlabel('Number of Neighbors (k)')
plt.tight_layout()
plt.show()

#Re-fit with best k
best_knn = knn(n_neighbors=(mean_acc.argmax()+1)).fit(X_train, y_train)
y_hat_knn = best_knn.predict(X_test)

#Accuracy 
print('Best KNN accuracy: ', mean_acc.max(), 'with k = ', mean_acc.argmax()+1)
print('KNN Jaccard Score: ', jaccard_score(y_test, y_hat_knn, average='weighted'))
print('KNN F1 Score: ', f1_score(y_test, y_hat_knn, average='weighted'))

## Decision Tree

num_trees = 10
tree_acc = np.zeros(num_trees-1)
for dp in range(1, num_trees):
    loan_tree = DecisionTreeClassifier(criterion = 'entropy', max_depth=dp)
    loan_tree.fit(X_train, y_train)
    predTree = loan_tree.predict(X_test)
    tree_acc[dp-1] = metrics.accuracy_score(y_test, predTree)

print('Best Decision Tree Accuracy: ', tree_acc.max())
best_tree = DecisionTreeClassifier(criterion= 'entropy', max_depth=tree_acc.argmax()+1)
best_tree.fit(X_train,y_train)
predTree=best_tree.predict(X_test)
tree.plot_tree(best_tree)
plt.title('Decision Tree for Loan Approval')
plt.show()

#Accuracy
print('X[4] is the Weekend Feature')
print('Tree Jaccard Score: ',jaccard_score(y_test, predTree, average='weighted'))
print('Tree F1 Score: ', f1_score(y_test, predTree, average='weighted'))


## Support Vector Machine rbf kernal
# Model
clf = svm.SVC(kernel='rbf')
clf.fit(X_train, y_train)
y_hat_rbf = clf.predict(X_test)

#Confusion Matrix
cnf_matrix = confusion_matrix(y_test, y_hat_rbf, labels=['PAIDOFF','COLLECTION'])
np.set_printoptions(precision=2)

ax = plt.subplot()
sns.heatmap(cnf_matrix, annot=True, fmt='g', cmap='Blues', cbar=False, ax=ax)

ax.set_xlabel('Predicted Labels'); ax.set_ylabel('True Labels')
ax.set_title('RBF SVM Confusion Matrix')
ax.xaxis.set_ticklabels(['Paidoff','Collection'])
ax.yaxis.set_ticklabels(['Paidoff','Collection'])
plt.show()

#Accuracy Scores
print('Jaccard Score, RBF SVM: ', jaccard_score(y_test, y_hat_rbf, average='weighted'))
print('F1 Score, RBF SVM: ', f1_score(y_test, y_hat_rbf, average='weighted'))
# print('F1 Score, RBF SVM: ', f1_score(y_test, y_hat_rbf, pos_label='PAIDOFF' ))

##Support Vector Machine linear kernal
#Model
clf2 = svm.SVC(kernel='linear')
clf2.fit(X_train, y_train)
y_hat_line = clf2.predict(X_test)

#Confusion Matrix 
cnf2_matrix = confusion_matrix(y_test, y_hat_line, labels=['PAIDOFF','COLLECTION'])
ax = plt.subplot()
sns.heatmap(cnf2_matrix, annot=True, fmt='g', cmap='Blues', cbar=False, ax=ax)

ax.set_xlabel('Predicted Labels'); ax.set_ylabel('True Labels')
ax.set_title('Linear SVM Confusion Matrix')
ax.xaxis.set_ticklabels(['Paidoff','Collection'])
ax.yaxis.set_ticklabels(['Paidoff','Collection'])
plt.show()

print('Jaccard Index, Linear SVM: ',jaccard_score(y_test, y_hat_line, average='weighted'))
print('F1 Score, Linear SVM: ', f1_score(y_test, y_hat_line, average='weighted'))
# print('F1 Score, Linear SVM: ', f1_score(y_test, y_hat_line, pos_label='PAIDOFF' ))


##Logistic Regression
#Model
LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train, y_train)
y_hat_lr = LR.predict(X_test)
y_har_lr_prob = LR.predict_proba(X_test)

#Confusion Matrix
ax = plt.subplot()
sns.heatmap(cnf_matrix, annot=True, fmt='g', cmap='Blues', cbar=False, ax=ax)

ax.set_xlabel('Predicted Labels'); ax.set_ylabel('True Labels')
ax.set_title('Logistic Regression Confusion Matrix')
ax.xaxis.set_ticklabels(['Paidoff','Collection'])
ax.yaxis.set_ticklabels(['Paidoff','Collection'])
plt.show() 

#Accuracy Scores
print('Jaccard Score, Logistic Regression: ', jaccard_score(y_test, y_hat_lr, average='weighted'))
print('F1 Score, Logistic Regression: ', f1_score(y_test, y_hat_lr, average='weighted'))

status_dict = {'COLLECTION':0, 'PAIDOFF':1}
y_test_bin = [status_dict[i] for i in y_test]
y_hat_lr_bin = [status_dict[i] for i in y_hat_lr]

print('Log Loss: ', log_loss(y_test_bin, y_hat_lr_bin))


# Model Evaluation
print('----------------------')
print('---Model Evaluation---')
print('----------------------')
#Preprocess Test Data
df_test = pd.read_csv('loan_test.csv')
df_test['due_date'] = pd.to_datetime(df_test['due_date'])
df_test['effective_date'] = pd.to_datetime(df_test['effective_date'])
df_test['education'].replace('Bechalor', 'Bachelor',inplace=True) #Fix spelling issue
df_test['dayofweek'] = df_test['effective_date'].dt.dayofweek
df_test['weekend']= df_test['dayofweek'].apply(lambda x: 1 if (x>3) else 0)

df_test.groupby(['Gender'])['loan_status'].value_counts(normalize=True)
df_test['Gender'].replace(to_replace=['male','female'], value=[0,1], inplace=True)

df_test.groupby(['education'])['loan_status'].value_counts(normalize=True)
df_test['education'].value_counts()
testFeat = df_test[['Principal','terms','age','Gender','weekend']]
testFeat = pd.concat([testFeat, pd.get_dummies(df_test['education'])],axis=1)
testFeat.drop(['Master or Above'], axis = 1, inplace=True) #only 2 instances

X_test_2 = testFeat
X_test_2 = preprocessing.StandardScaler().fit(X_test_2).transform(X_test_2)
# X_retrain = Feature
# y_retrain = df_loan['loan_status'].values

y_test_2 = df_test['loan_status'].values

# KNN Evaluation
y_hat_knn_2 = best_knn.predict(X_test_2)
print('KNN Jaccard Score: ', jaccard_score(y_test_2, y_hat_knn_2, average='weighted'))
print('KNN F1 Score: ', f1_score(y_test_2, y_hat_knn_2, average='weighted'))

# Decision Tree
test_tree_pred = best_tree.predict(X_test_2)
print('Tree Jaccard Score: ',jaccard_score(y_test_2, test_tree_pred, average='weighted'))
print('Tree F1 Score: ', f1_score(y_test_2, test_tree_pred, average='weighted'))

# SVM 
y_hat_rbf_2 = clf.predict(X_test_2)
print('Jaccard Score, RBF SVM: ', jaccard_score(y_test_2, y_hat_rbf_2, average='weighted'))
print('F1 Score, RBF SVM: ', f1_score(y_test_2, y_hat_rbf_2, average='weighted'))

y_hat_line_2 = clf2.predict(X_test_2)
print('Jaccard Index, Linear SVM: ',jaccard_score(y_test_2, y_hat_line_2, average='weighted'))
print('F1 Score, Linear SVM: ', f1_score(y_test_2, y_hat_line_2, average='weighted'))

# Logistic Regression

y_hat_lr_2 = LR.predict(X_test_2)

print('Jaccard Score, Logistic Regression: ', jaccard_score(y_test_2, y_hat_lr_2, average='weighted'))
print('F1 Score, Logistic Regression: ', f1_score(y_test_2, y_hat_lr_2, average='weighted'))
y_test_2 = [status_dict[i] for i in y_test_2]
y_hat_lr_2 = [status_dict[i] for i in y_hat_lr_2]
print('Log Loss: ', log_loss(y_test_2, y_hat_lr_2))