# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 09:53:51 2022

@author: Nick
"""

import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
from sklearn import linear_model
from sklearn.metrics import r2_score

path = r'C:\Users\1109336\Documents\Python\Coursera\MachineLearningPython\FuelConsumptionCo2.csv'
#path = r'C:\Users\1109336\Nick\Python\Coursera\MachineLearningPython\FuelConsumptionCo2.csv'
df = pd.read_csv(path)
    
cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
cdf.hist()
plt.show()

plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS, color='blue')
plt.xlabel('Fuel Consmption Comb')
plt.ylabel('Emissions')
plt.show()

plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS, color='blue')
plt.xlabel('Engine Size')
plt.ylabel('Emissions')
plt.show()

plt.scatter(cdf.CYLINDERS, cdf.CO2EMISSIONS, color='blue')
plt.xlabel('Cylinders')
plt.ylabel('Emissions')
plt.show()

# Splitting Data
msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]

# Training with Engine Size
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color='blue')
plt.xlabel('Engine Size')
plt.ylabel('Emission')
plt.show()

regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])

regr.fit(train_x,train_y)
print('Coefficients: ', regr.coef_)
print('Intercept: ', regr.intercept_)

plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color='blue')
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')
plt.xlabel('Engine Size')
plt.ylabel('Emission')
plt.title('Engine Size vs. Emission, Linear Regression')
plt.show()

#Evaluation
test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
test_y_ = regr.predict(test_x)

print('Mean absolute error: %.2f' % np.mean(np.absolute(test_y_ - test_y)))
print('Residual sum of squares (MSE): %.2f' % np.mean((test_y_ - test_y) ** 2))
print('R2-score: %.2f' % r2_score(test_y , test_y_) )

# Train with Fuel Consumption
train_x_fuel = np.asanyarray(train[['FUELCONSUMPTION_COMB']])
test_x_fuel = np.asanyarray(test[['FUELCONSUMPTION_COMB']])

regr_fuel = linear_model.LinearRegression()
regr_fuel.fit(train_x_fuel,train_y)

y_hat_fuel = regr_fuel.predict(test_x_fuel)

print('Mean absolute error: %.2f' % np.mean(np.absolute(y_hat_fuel - test_y)))
print('Residual sum of squares (MSE): %.2f' % np.mean((y_hat_fuel - test_y) ** 2))
print('R2-score: %.2f' % r2_score(y_hat_fuel, test_y_) )