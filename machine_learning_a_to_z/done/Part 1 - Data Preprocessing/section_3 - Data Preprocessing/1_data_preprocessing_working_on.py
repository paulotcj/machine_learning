# Data Preprocessing Template

print('Importing the libraries')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

print('Importing the dataset')
dataset = pd.read_csv('./Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

print(x)
print('----')
print(y)


# we should not split here yet
# print('Splitting the dataset into the Training set and Test set')
# from sklearn.model_selection import train_test_split
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

# print('x_train')
# print(x_train)
# print('----')
# print('x_test')
# print(x_test)
# print('----')
# print('y_train')
# print(y_train)
# print('----')
# print('y_test')
# print(y_test)

print('----------------------------------------------')

print('Taking care of missing data')
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3]) #we are targeting the columns 1 and 2 (age, and salary)



#----------------------------------------------
print('Encoding categorical data')
print('Encoding the Independent Variable')

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# The transformers argument is a list of tuples, where each tuple defines a transformation 
# to apply to specific columns. In this case, there's only one tuple: 
#     ('encoder', OneHotEncoder(), [0]).
# The first element of the tuple, 'encoder', is a name for this transformer (you can choose 
# any name you like).
# The second element, OneHotEncoder(), is an instance of the transformer to be used. 
# OneHotEncoder is a class from sklearn.preprocessing that's used to convert categorical 
# data into a binary vector format.
# The third element, [0], is a list of indices of the columns to which this transformer 
# should be applied. In this case, the transformer will be applied to the first column 
# of the input dataset (since Python uses 0-based indexing).

# The remainder argument specifies what to do with the columns that are not explicitly 
# selected for transformation. In this case, 'passthrough' means that all columns not 
# specified in the transformers list will be left as they are in the output dataset.

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])] , remainder='passthrough')
x = np.array( ct.fit_transform(x) )

print(x)
#----------------------------------------------


print('----------------------------------------------')
print('Encoding the Dependent Variable')

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

print(y)

print('----------------------------------------------')
print('Splitting the dataset into the Training set and Test set')
print('Note: do not scale before splitting, as this will cause data leakage')
from sklearn.model_selection import train_test_split

# If an integer is passed, random_state will use it as a seed to the random number 
# generator. This ensures that the random numbers are generated in the same order.
#
# For example, if you always want the same train/test data split, you can set 
# random_state to an integer value. This can be useful for debugging, so that 
# your results are reproducible.

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1)

print('x_train - the data we have and we will be doing the training on this')
print(x_train)
print('----')
print('x_test - the data we will be testing on, to be used as input later')
print(x_test)
print('----')
print('y_train - the results we have and we will be doing the training on this')
print(y_train)
print('----')
print('y_test - the results we will be testing against, to be used as input later')
print(y_test)

print('----------------------------------------------')

print('Feature Scaling')
#Normalization: x_norm = ( x - min(x) ) / ( max(x) - min(x) )
#  good when you don't have a normal distribution or when you are unsure if you have one
#  the range of the values will be between 0 and 1


#Standardization: x_stand = ( x - mean(x) ) / standard_deviation(x)
#  "often a safe choice and works well in many cases. It's less sensitive to outliers compared to normalization, and it can be applied to a 
#   wider range of datasets without assumptions about the distribution of the features."
#  roughly the range can be expected to be between -3 and 3 (but realistically it can be any value)



print('Normalization - MinMaxScaler')
from sklearn.preprocessing import MinMaxScaler
min_max_scaler = MinMaxScaler()
x_train
x_train[:, 3:] = min_max_scaler.fit_transform(x_train[:, 3:])
x_test[:, 3:] = min_max_scaler.transform(x_test[:, 3:]) # we do not fit the test set, we only transform it

print('x_train - after scaling')
print(x_train)
print('----')
print('x_test - after scaling')
print(x_test)


print('----------------------------------------------')
print('Standardization - StandardScaler')
from sklearn.preprocessing import StandardScaler
standard_scaler = StandardScaler()

# [:, 3:] -> [all rows, columns starting from 3 to the end]
x_train[:, 3:] = standard_scaler.fit_transform(x_train[:, 3:]) #This means it calculates the mean and standard deviation of the data, then subtracts the mean and divides by the standard deviation for each value.
x_test[:, 3:] = standard_scaler.transform(x_test[:, 3:]) #transforms the x_test data using the mean and standard deviation calculated from the x_train data. This is done to avoid data leakage, where information from the test set is used to scale the training set.

print('x_train - after scaling')
print(x_train)
print('----')
print('x_test - after scaling')
print(x_test)