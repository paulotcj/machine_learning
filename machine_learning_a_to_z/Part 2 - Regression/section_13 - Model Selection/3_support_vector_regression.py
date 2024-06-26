print('----------------------------------------------')
print('Support Vector Regression (SVR)')

print('----------------------------------------------')
print('Import the libraries')
import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd

print('----------------------------------------------')
print('Import the dataset')
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
y = y.reshape(len(y),1)

print('----------------------------------------------')
print('Split the dataset into the Training set and Test set')
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

print('----------------------------------------------')
print('Feature Scaling')
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
x_train = sc_X.fit_transform(x_train)
y_train = sc_y.fit_transform(y_train)

print('----------------------------------------------')
print('Train the SVR model on the Training set')
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(x_train, y_train)

print('----------------------------------------------')
print('Predict the Test set results')
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(x_test)).reshape(-1,1))
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

print('----------------------------------------------')
print('Evaluate the Model Performance')
from sklearn.metrics import r2_score
print(r2_score(y_test, y_pred))