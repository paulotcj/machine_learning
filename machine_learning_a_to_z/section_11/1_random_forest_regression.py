print('Random Forest Regression')

print('Import the libraries')

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

print('----------------------------------------------')

print('mport the dataset')

dataset = pd.read_csv('./Position_Salaries.csv')
x = dataset.iloc[:,1:-1].values #all rows, starting from the second column minus the last column
y = dataset.iloc[:,-1].values #all rows, last column


# print('----------------------------------------------')

# print('----------------------------------------------')