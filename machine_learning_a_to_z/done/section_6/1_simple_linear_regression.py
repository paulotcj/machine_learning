print('Simple Linear Regression')

print('Import the libraries')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


print('----------------------------------------------')
print('Import the dataset')
dataset = pd.read_csv('./Salary_Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
print('X - Years of experience')
print(x)
print('----')
print('y - Salary')
print(y)


print('----------------------------------------------')
print('Split the dataset into the Training set and Test set')
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
print('X_train - Training set Years of experience')
print(x_train)
print('----')
print('X_test - Test set Years of experience')
print(x_test)
print('----')
print('y_train - Training set Salary')
print(y_train)
print('----')
print('y_test - Test set Salary')
print(y_test)
print('----------------------------------------------')

print('Simple Linear Regression model on the Training set')
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train) #x_train is the years of experience, y_train is the salary
print('----------------------------------------------')

print('Predicting the Test set results')
y_pred = regressor.predict(x_test) #y_pred is the predicted salary
print('y_pred - Predicted salary. Note y_pred should be close to/predict y_test.')
print(y_pred)
print('----------------------------------------------')

print('Visualising the Training set results')
plt.scatter(x_train, y_train, color = 'red') #this will be points using the data from the training set
plt.plot(x_train, regressor.predict(x_train), color = 'blue') #this will be the line of best fit using the data from the training set

#labels
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')

plt.show()
print('----------------------------------------------')

print('Visualising the Test set results')
#Now we compare the predicted salary to the actual salary
plt.scatter(x_test, y_test, color = 'red')
plt.scatter(x_test, y_pred, color='green', marker='+')


plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.text(0.05, 0.9, 'red dots = actual data, green crosses = predicted data', transform=plt.gca().transAxes)
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()