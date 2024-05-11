print('Decision Tree Regression')

print('Import the libraries')

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

print('----------------------------------------------')

print('mport the dataset')

dataset = pd.read_csv('./Position_Salaries.csv')
x = dataset.iloc[:,1:-1].values #all rows, starting from the second column minus the last column
y = dataset.iloc[:,-1].values #all rows, last column


print('----------------------------------------------')

print('Train the Decision Tree Regression model on the whole dataset')


from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(x,y)

#random_state = 0 -> "determines the random number generation used to create 
#   the decision tree. When you set a specific random_state, the decision tree 
#   will be generated using the same random number generation each time, which 
#   can be useful for reproducibility."

# "For example, if you are training a decision tree model and want to be able 
#   to reproduce the results exactly at a later time, you can set the 
#   random_state to a fixed value. This will ensure that the model is generated 
#   in the same way each time, which can be useful for debugging or for comparing 
#   different models."

# "If you do not specify a random_state, the decision tree will be generated using 
#    a different random number generation each time, which can lead to slightly 
#    different results each time the model is trained. This can be useful for 
#    certain types of experimentation, but it can make it more difficult to reproduce 
#    results."
# https://www.kaggle.com/discussions/questions-and-answers/109268

print('----------------------------------------------')

print('Predicting a new result')
predict_from_this = 6.5
target_to_confirm = 160_000
print(f'    predict_from_this: {predict_from_this}')
print(f'    target_to_confirm: {target_to_confirm}')


predicted_result = regressor.predict([[predict_from_this]])

print('predicted_result:')
print(predicted_result)