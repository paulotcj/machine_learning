print('Recurrent Neural Network')
print('----------------------------------------------')

print('Part 1 - Data Preprocessing')
print('----------------------------------------------')

print('import the libraries')

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

print('----------------------------------------------')
print('import the training set\n')
str_path = './deep_learning_a_to_z\part_3\section_10_RNN/'
dataset_train = pd.read_csv( f'{str_path}Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values

print(f'first 10 rows of training_set: \n{training_set[0:10]}')
# print(training_set[0:10])

print('----------------------------------------------')
print('feature scaling\n')
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
training_set_scaled = sc.fit_transform(training_set)
print(f'first 10 rows of training_set_scaled: \n{training_set_scaled[0:10]}')

print(f'len of training_set_scaled: {len(training_set_scaled)}')

print('----------------------------------------------')
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
print('\n\n')

print('----------------------------------------------')
print('reshape')

print(f'x_train.shape: {x_train.shape}')
x_train = np.reshape( x_train , (x_train.shape[0], x_train.shape[1], 1) ) #just adding another dimension as required by the RNN
print(f'x_train.shape: {x_train.shape}')
# print(f'first 10 rows of x_train: \n{x_train[0:10]}')

print('----------------------------------------------')
print('Part 2 - Building the RNN')


print('----------------------------------------------')
print('Importing the Keras libraries and packages')

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

print('----------------------------------------------')

print('Initialising the RNN')
regressor = Sequential() #keras model

print('----------------------------------------------')
print('Adding the first LSTM layer and some Dropout regularisation')
# here we are defining the first LSTM layer with 50 LSTM units in this layer.
# Important, return_sequences = True, allows the subsequent LSTM layers to receive the same full sequence as input, and not just the last element 
#  of the sequence.
regressor.add( LSTM( units = 50, return_sequences = True , input_shape = ( x_train.shape[1], 1) ) ) #x_train.shape[1] is supposed to be 60

# Dropout: "This process helps in making the model less sensitive to the specific weights of neurons, leading to a more generalized model that 
#  performs better on unseen data."
# 0.2 Dropout: the model is will randomly ignore 20% of the outputs from the previous layer during each training iteration. The goal is to reduce overfitting
regressor.add( Dropout(0.2) )

#----
print('Adding a second LSTM layer and some Dropout regularisation')
regressor.add( LSTM( units = 50 , return_sequences = True)  )
regressor.add( Dropout(0.2) )

#----
print('Adding a third LSTM layer and some Dropout regularisation')
regressor.add( LSTM( units = 50 , return_sequences = True ) )
regressor.add( Dropout(0.2) )

print('Adding a fourth LSTM layer and some Dropout regularisation')
regressor.add( LSTM( units = 50) )
regressor.add( Dropout(0.2) )

print('Adding the output layer')
regressor.add( Dense( units = 1 ))

print('----------------------------------------------')
print('Compiling the RNN')
regressor.compile( optimizer = 'adam', loss = 'mean_squared_error' )

print('----------------------------------------------')
print('Fitting the RNN to the Training set')
regressor.fit( x_train , y_train , epochs = 100 , batch_size = 32 )


print('----------------------------------------------')
print('Part 3 - Making the predictions and visualising the results')

print('----------------------------------------------')
print('Getting the real stock price of 2017')
dataset_test = pd.read_csv(f'{str_path}Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

print('----------------------------------------------')
print('Getting the predicted stock price of 2017')
dataset_total = pd.concat( (dataset_train['Open'], dataset_test['Open']), axis = 0 )
#--------------------------------

# Part 3 - Making the predictions and visualising the results

# Getting the real stock price of 2017
# dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
# real_stock_price = dataset_test.iloc[:, 1:2].values

# Getting the predicted stock price of 2017
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualising the results
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()