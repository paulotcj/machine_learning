print('Multiple Linear Regression')

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

print('----------------------------------------------')

print('import dataset')

dataset = pd.read_csv('./50_Startups.csv')
x = dataset.iloc[:, : -1].values #all the rows and all the columns except the last one
y = dataset.iloc[:, -1].values #all the rows but only the last column (Profit)
print('independent variables')
print(x)
print('----')
print('dependent variables')
print(y)
print('----------------------------------------------')

print('encode categorical data')

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# The transformers argument is a list of tuples, where each tuple defines a transformation 
# to apply to specific columns. In this case, there's only one tuple: 
#     ('encoder', OneHotEncoder(), [3]).
# The first element of the tuple, 'encoder', is a name for this transformer (you can choose 
# any name you like).
# The second element, OneHotEncoder(), is an instance of the transformer to be used. 
# OneHotEncoder is a class from sklearn.preprocessing that's used to convert categorical 
# data into a binary vector format.
# The third element, [3], is a list of indices of the columns to which this transformer 
# should be applied. In this case, the transformer will be applied to the first column 
# of the input dataset (since Python uses 0-based indexing).

# The remainder argument specifies what to do with the columns that are not explicitly 
# selected for transformation. In this case, 'passthrough' means that all columns not 
# specified in the transformers list will be left as they are in the output dataset.

col_transformer = ColumnTransformer(transformers=[('made-up-name-whatever-encoder', OneHotEncoder(), [3])], remainder = 'passthrough')
x = np.array(col_transformer.fit_transform(x))
print(x)
print('----------------------------------------------')


print('split the dataset into the Training set and Test set')

from sklearn.model_selection import train_test_split

#note abouyt random_state= 0 -> typically we like to see random samples, this helps the model to be more consistent.
# however, if we want to reproduce the same results, we can set random_state = 1
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state= 0)

print('x_train')
print(x_train)
print('----')
print('x_test')
print(x_test)
print('----')
print('y_train')
print(y_train)
print('----')
print('y_test')
print(y_test)
print('----------------------------------------------')


print ('Setting up the Multiple Linear Regression model for this Training set')


from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(x_train, y_train)

print('----------------------------------------------')


print('predicting the test set results')
print('  note: y_pred should be close to y_test') 

y_pred = regressor.predict(x_test)
np.set_printoptions(precision = 2)

# Convert the y_pred array into a two-dimensional array with one column.
# The len(y_pred) function is used to determine the number of rows for the reshaped array. 
#  len(y_pred) returns the number of elements in the y_pred array, which will be the number 
#  of rows in the reshaped array.
# The 1 passed as the second argument to the reshape function specifies that there should be 
#  one column in the reshaped array.

# print(y_pred)
# print('----')
# print(y_pred.reshape(len(y_pred),1))


# The numpy.concatenate() function is used to join two or more arrays of the same shape along 
#   a specified axis. The axis along which the arrays will be joined is defined by the second 
#   argument. If axis is None, arrays are flattened before use. In this case, the axis is 
#   set to 1, which means the function will concatenate the arrays along the columns.

# Before the arrays are concatenated, they are reshaped using the numpy.reshape() function. 
#   This function gives a new shape to an array without changing its data. Here, it is used to 
#   convert the y_pred and y_test arrays into two-dimensional arrays with one column. The 
#   len(y_pred) and len(y_test) functions are used to determine the number of rows for the 
#   reshaped arrays. They return the number of elements in the y_pred and y_test arrays, which 
#   will be the number of rows in the reshaped arrays.

# So, y_pred.reshape(len(y_pred),1) will return a new array that has the same data as y_pred, 
#   but structured as a two-dimensional array with len(y_pred) rows and 1 column. Similarly, 
#   y_test.reshape(len(y_test),1) will return a new array that has the same data as y_test, 
#   but structured as a two-dimensional array with len(y_test) rows and 1 column.

# Finally, np.concatenate(( y_pred.reshape(len(y_pred),1) , y_test.reshape(len(y_test),1) ),1) 
#   will concatenate these two arrays along the columns and return a new array. This is often 
#   done in machine learning tasks when you want to compare the predicted and actual values 
#   side by side.

print( np.concatenate(( y_pred.reshape(len(y_pred),1) , y_test.reshape(len(y_test),1) ),1) )








