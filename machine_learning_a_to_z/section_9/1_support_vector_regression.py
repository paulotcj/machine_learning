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
print('  for the SVR model, we need to apply feature scaling')

#Normalization: x_norm = ( x - min(x) ) / ( max(x) - min(x) )
#  good when you don't have a normal distribution or when you are unsure if you have one
#  the range of the values will be between 0 and 1


#Standardization: x_stand = ( x - mean(x) ) / standard_deviation(x)
#  "often a safe choice and works well in many cases. It's less sensitive to outliers compared to normalization, and it can be applied to a 
#   wider range of datasets without assumptions about the distribution of the features."
#  roughly the range can be expected to be between -3 and 3 (but realistically it can be any value)
from sklearn.preprocessing import StandardScaler
standard_scaler_x = StandardScaler()
standard_scaler_y = StandardScaler()

x = standard_scaler_x.fit_transform(x)
y = standard_scaler_y.fit_transform(y)

print('x:')
print(x)
print('----')
print('y:')
print(y)
print('----------------------------------------------')

print('Trai the SVR model on the whole dataset')

# (RBF) Radial Basis Function. The RBF kernel is a popular choice because it is able to 
#   create non-linear decision boundaries and can model complex relationships 
#   between the input features and the target variable.

from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(x,y)

print('----------------------------------------------')

print('Predicting a new result')
predict_from_this = 6.5
target_to_confirm = 160_000
print(f'    predict_from_this: {predict_from_this}')
print(f'    target_to_confirm: {target_to_confirm}')

result = standard_scaler_y.inverse_transform( #transform the fitted value back to the original scale
        regressor.predict(  #predict the value
            standard_scaler_x.transform([[predict_from_this]]) #transform the input value to the scale used in the model
        ).reshape(-1,1) # -1 rows means as many rows as needed, and 1 column
    )
print('prediction:')
print(result)




