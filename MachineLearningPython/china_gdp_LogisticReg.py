# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 10:17:14 2022

@author: 1109336
"""
import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from scipy.optimize import curve_fit

path = r'C:\Users\1109336\Documents\Python\Coursera\MachineLearningPython\china_gdp.csv'
#path = r'C:\Users\Nick\Documents\Python\Coursera\MachineLearningPython\china_gdp.csv'
df = pd.read_csv(path)
#df.head(10)
df.tail()


plt.figure(figsize=(8,5))
x_data, y_data = (df["Year"].values, df["Value"].values)
plt.plot(x_data, y_data, 'ro')
plt.ylabel('GDP')
plt.xlabel('Year')
plt.title('China GDP [1960-2014]')
plt.show()

def sigmoid(x, Beta_1, Beta_2):
    y = 1 / (1 +np.exp(-Beta_1*(x-Beta_2)))
    return y

beta_1 = 0.10
beta_2 = 1990.0

Y_pred = sigmoid(x_data, beta_1, beta_2)

plt.plot(x_data, Y_pred*15000000000000)
plt.plot(x_data, y_data, 'ro')

x_norm = x_data/max(x_data)
y_norm = y_data/max(y_data)

popt, pcov = curve_fit(sigmoid, x_norm, y_norm)
print('beta_1 = %f, beta_2 = %f' % (popt[0], popt[1]))


x = np.linspace(1960, 2015, 55)
x = x/max(x)
plt.figure(figsize=(8,5))
y = sigmoid(x, *popt)
plt.plot(x_norm, y_norm, 'ro', label='data')
plt.plot(x,y, linewidth = 3.0, label='fit')
plt.legend(loc='best')
plt.ylabel('GDP')
plt.xlabel('Year')
plt.title('China GDP Logistic Fit [1960-2014]')
plt.show()

# let's actually model it: 
# msk = np.random.rand(len(df)) < 0.8
# train = df[msk]
# test = df[~msk]

# x_train = train[['Year']]
# y_train = train[['Value']]

# x_test = test[['Year']]
# y_test = test[['Value']]

# poly = PolynomialFeatures(degree = 2)
# X_poly = poly.fit_transform(x_train)

# lr = linear_model.LogisticRegression()
# train_y_poly = lr.fit(X_poly,y_train)

# test_x_poly = poly.transform(x_test)
# test_y_poly = lr.predict(test_x_poly)

# print('MAE: %.2f' % np.mean(np.absolute(test_y_poly - y_test)))
# print('MSE: %.2f' % np.mean((test_y_poly - y_test) ** 2))
# print('R2-Score: %.2f' % r2_score(y_test, test_y_poly))

#IBM Module Model: 
msk = np.random.rand(len(df))< 0.8
train_x = x_norm[msk]
train_y = y_norm[msk]
test_x = x_norm[~msk]
test_y = y_norm[~msk]

popt, pcov = curve_fit(sigmoid, train_x, train_y)
y_hat = sigmoid(test_x, *popt)

print('MAE: %.2f' % np.mean(np.absolute(y_hat - test_y)))
print('MSE: %.2f' % np.mean((y_hat - test_y) ** 2))
print('R2-Score: %.2f' % r2_score(test_y, y_hat))
