print('Polynomial Regression')

# Importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

print('----------------------------------------------')
print('Importing the dataset')

dataset = pd.read_csv('./Position_Salaries.csv')
x = dataset.iloc[:, 1:-1 ].values #the first column contains the positions names, and we don't need it
y = dataset.iloc[:, -1].values #the last column contains the salaries

print('x:')
print(x)
print('----')
print('y:')
print(y)
print('----------------------------------------------')


print('Making a Linear Regression model with the whole dataset')
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x, y) # x = training data, y = target date

print('----------------------------------------------')

print('Making the Polynomial Regression model with the whole dataset')

from sklearn.preprocessing import PolynomialFeatures

#polynomial regression: y = b0 + b1*x1 + b2*x1^2 + b3*x1^3 + ... + bn*x1^n
# so the degree is the number of x1^i terms in the equation - the higher the degree, 
# the more complex the model
poly_reg = PolynomialFeatures(degree = 5) #we should use 4 for a good fit


x_poly = poly_reg.fit_transform(x)
lin_reg_poly = LinearRegression()
lin_reg_poly.fit(x_poly, y) # x = training data, y = target date

print('x_poly:')
print(x_poly)




print('----------------------------------------------')

print('Creating the visualisation for the Linear Regression model')

plt.scatter(x,y, color = 'red')
plt.plot(x, lin_reg.predict(x), color = 'blue')

plt.title('Truth or Bluff (Linear Regression)')

plt.xlabel('Position Level')
plt.ylabel('Salary')

plt.show()

print('----------------------------------------------')

print('Creating the visualisation for the Polynomial Regression model')
plt.scatter(x, y, color = 'red')
plt.plot(x, lin_reg_poly.predict( poly_reg.fit_transform(x) ) , color = 'blue')





plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

print('----------------------------------------------')


# print('Visualising the Polynomial Regression results (for higher resolution and smoother curve)')
# x_grid = np.arange(min(x), max(x), 0.1)
# x_grid = x_grid.reshape((len(x_grid), 1))

# plt.scatter(x, y, color = 'red')
# plt.plot(x_grid, lin_reg_poly.predict(poly_reg.fit_transform(x_grid)), color = 'blue')
# plt.title('Truth or Bluff (Polynomial Regression)')
# plt.xlabel('Position level')
# plt.ylabel('Salary')
# plt.show()
