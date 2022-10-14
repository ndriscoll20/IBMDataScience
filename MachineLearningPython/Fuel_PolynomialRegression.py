# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 08:14:23 2022

@author: Nick
"""
import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

path = r'C:\Users\1109336\Documents\Python\Coursera\MachineLearningPython\FuelConsumptionCo2.csv'
#path = r'C:\Users\Nick\Documents\Python\Coursera\MachineLearningPython\FuelConsumptionCo2.csv'
df = pd.read_csv(path)
#df.head()

cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS','FUELCONSUMPTION_COMB']]
#cdf2 = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY']]

# Split the data
msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]

train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])

test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])

# Train the Model
poly = PolynomialFeatures(degree=2)
train_x_poly = poly.fit_transform(train_x)
# Output the polynomial fit matrix:
# train_x_poly

clf = linear_model.LinearRegression()
train_y_ = clf.fit(train_x_poly, train_y)

print('Coeffcients: ', clf.coef_)
print('Intercept: ', clf.intercept_)

plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color='blue')
XX = np.arange(0.0, 10.0, 0.1)
yy = clf.intercept_[0] + clf.coef_[0][1]*XX + clf.coef_[0][2]*np.power(XX,2)
plt.plot(XX, yy, '-r')
plt.xlabel('Engine Size')
plt.ylabel('Co2 Emissions')
plt.title('Polynomial Fit (2nd Degree)')
plt.show()

#Evaluate the Accuracy
test_x_poly = poly.transform(test_x)
test_y_ = clf.predict(test_x_poly)

print('MAE: %.2f' % np.mean(np.absolute(test_y_ - test_y)))
print('MSE: %.2f' % np.mean((test_y_ - test_y) ** 2))
print('R2-Score: %.2f' % r2_score(test_y, test_y_))

# Cubic 
poly = PolynomialFeatures(degree=3)
train_x_poly3 = poly.fit_transform(train_x)

clf3 = linear_model.LinearRegression()
train_y_3 = clf3.fit(train_x_poly3, train_y)

print('Coeffcients: ', clf3.coef_)
print('Intercept: ', clf3.intercept_)

plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color='blue')
yy3 = clf3.intercept_[0] + clf3.coef_[0][1]*XX + clf3.coef_[0][2]*np.power(XX,2) + clf3.coef_[0][3] * np.power(XX,3)

plt.plot(XX, yy3, '-r')
plt.xlabel('Engine Size')
plt.ylabel('Co2 Emissions')
plt.title('Polynomial Fit (Cubic)')
plt.show()

#Evaluate
test_x_poly3 = poly.transform(test_x)
test_y_3 = clf3.predict(test_x_poly3)

print('MAE: %.2f' % np.mean(np.absolute(test_y_3 - test_y)))
print('MSE: %.2f' % np.mean((test_y_3 - test_y) ** 2))
print('R2-Score: %.2f' % r2_score(test_y, test_y_3))