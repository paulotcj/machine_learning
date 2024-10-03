print('----------------------------------------------')
print('Random Forest Regression')
print('----------------------------------------------')
print('Import the libraries')

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

print('----------------------------------------------')

print('Import the dataset')

dataset = pd.read_csv('./Position_Salaries.csv')
x = dataset.iloc[:,1:-1].values #all rows, starting from the second column minus the last column
y = dataset.iloc[:,-1].values #all rows, last column


print('----------------------------------------------')

print('Train the Random Forest Regression model on the whole dataset')
from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(x, y)

# "the random algorithm will be used in any case. Passing any value (whether 
#   a specific int, e.g., 0, or a RandomState instance), will not change that. 
#   The only rationale for passing in an int value (0 or otherwise) is to make 
#   the outcome consistent across calls: if you call this with random_state=0 
#   (or any other value), then each and every time, you'll get the same result."
# https://stackoverflow.com/questions/39158003/confused-about-random-state-in-decision-tree-of-scikit-learn

print('----------------------------------------------')

print('Predicting a new result')
predict_from_this = 6.5
target_to_confirm = 160_000
print(f'    predict_from_this: {predict_from_this}')
print(f'    target_to_confirm: {target_to_confirm}')



predicted_result = regressor.predict([[predict_from_this]])

print('predicted_result:')
print(predicted_result)

