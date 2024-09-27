print('Recurrent Neural Network\n\n')

print('Part 1 - Data Preprocessing\n\n')

print('import the libraries')

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

print('import the training set\n')
str_path = './deep_learning_a_to_z\part_3\section_10_RNN/'
dataset_train = pd.read_csv( f'{str_path}Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values

print(f'first 10 rows of training_set: \n{training_set[0:10]}')
# print(training_set[0:10])

print('feature scaling\n')
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
training_set_scaled = sc.fit_transform(training_set)
print(f'first 10 rows of training_set_scaled: \n{training_set_scaled[0:10]}')

print(f'len of training_set_scaled: {len(training_set_scaled)}')

print('creating a data structure with 60 timesteps and 1 output')
x_train = []
y_train = []
for i in range(60, len(training_set_scaled)): #startting i at 60 to offset the 60 timesteps slice
    #                                    60 - 60 = 0, 61 - 60 = 1, 62 - 60 = 2, ..., 
    x_train.append( training_set_scaled[ i - 60 : i, 0 ] ) #this means rows 0, to 59, 1 to 60, 2 to 61. And the column is 0
    y_train.append( training_set_scaled[ i , 0] ) #this means rows 60, 61, 62, ... and the column is 0

x_train, y_train = np.array(x_train), np.array(y_train)

print(f'first 10 rows of x_train: \n{x_train[0:10]}')
print(f'first 10 rows of y_train: \n{y_train[0:10]}')

print('reshape')

print(f'x_train.shape: {x_train.shape}')
x_train = np.reshape( x_train , (x_train.shape[0], x_train.shape[1], 1) ) #just adding another dimension as required by the RNN
print(f'x_train.shape: {x_train.shape}')
# print(f'first 10 rows of x_train: \n{x_train[0:10]}')

print('Part 2 - Building the RNN')
exit()

# Importing the Keras libraries and packages

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

exit()


#--------------------------------




# # Part 2 - Building the RNN

# # Importing the Keras libraries and packages
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.layers import LSTM
# from keras.layers import Dropout

# Initialising the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)