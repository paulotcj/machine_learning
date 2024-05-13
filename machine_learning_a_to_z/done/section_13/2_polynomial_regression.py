print('----------------------------------------------')
print('Polynomial Regression')
print('----------------------------------------------')
print('Import the libraries')
import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd

print('----------------------------------------------')
print('Import the dataset')
dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

print('----------------------------------------------')
print('Split the dataset into the Training set and Test set')
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

print('----------------------------------------------')
print('Train the Polynomial Regression model on the Training set')
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
poly_reg = PolynomialFeatures(degree = 5)
x_poly = poly_reg.fit_transform(x_train)
regressor = LinearRegression()
regressor.fit(x_poly, y_train)

print('----------------------------------------------')
print('Predict the Test set results')
y_pred = regressor.predict(poly_reg.transform(x_test))
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

print('----------------------------------------------')
print('Evaluate the Model Performance')
from sklearn.metrics import r2_score
print(r2_score(y_test, y_pred))