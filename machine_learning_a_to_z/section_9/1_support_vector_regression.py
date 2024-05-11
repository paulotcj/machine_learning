print('Support Vector Regression (SVR)')
print('----------------------------------------------')
print('Import libraries')

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
print('----------------------------------------------')

print('Import the dataset')
dataset = pd.read_csv('./Position_Salaries.csv')
x = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values

print('x:')
print(x)
print('----')
print('y:')
print(y)
print('----')
y = y.reshape(len(y), 1) #reshape y to a 2D array - y.len as rows and 1 column
print('y = y.reshape(len(y), 1):')
print(y)
print('----------------------------------------------')

print('Feature Scaling')

#Normalization: x_norm = ( x - min(x) ) / ( max(x) - min(x) )
#  good when you don't have a normal distribution or when you are unsure if you have one
#  the range of the values will be between 0 and 1


#Standardization: x_stand = ( x - mean(x) ) / standard_deviation(x)
#  "often a safe choice and works well in many cases. It's less sensitive to outliers compared to normalization, and it can be applied to a 
#   wider range of datasets without assumptions about the distribution of the features."
#  roughly the range can be expected to be between -3 and 3 (but realistically it can be any value)
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()

x = sc_x.fit_transform(x)
y = sc_y.fit_transform(y)

print('x:')
print(x)
print('----')
print('y:')
print(y)
print('----------------------------------------------')





