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
# str_path = './deep_learning_a_to_z\part_3\section_10_RNN/'
str_path = ''

# this stock price contains the range from 2012-01-03 to 2016-12-30
dataset_train = pd.read_csv( f'{str_path}Google_Stock_Price_Train.csv') #tabs: Date, Open, High, Low, Close, Volume
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
# regressor.fit( x_train , y_train , epochs = 100 , batch_size = 32 )


# Check if the model file exists
import os
from keras.models import load_model
model_path = 'rnn_model.h5'

if os.path.exists(model_path):
    print('Loading the existing model...')
    regressor = load_model(model_path)
else:
    print('Training the model...')
    regressor.fit(x_train, y_train, epochs=100, batch_size=32)
    print('Saving the model...')
    regressor.save(model_path)


print('----------------------------------------------')
print('Part 3 - Making the predictions and visualising the results')

print('----------------------------------------------')
print('Getting the real stock price of 2017')
dataset_test = pd.read_csv(f'{str_path}Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values #column 1, and all rows

print(f'first 10 rows of real_stock_price: \n{real_stock_price[0:10]}')


print('----------------------------------------------')
print('Getting the predicted stock price of 2017')

# dataset_train contains the stock price ranging from 2012-01-03 to 2016-12-30
#dataset_test contains the stock price ranging from 2017-01-03 to 2017-01-31 (1 month)
dataset_total = pd.concat( (dataset_train['Open'], dataset_test['Open']), axis = 0 ) #'Open' is the column name
print(f'first 10 rows of dataset_total: \n{dataset_total[0:10]}')

print('----------------------------------------------')
print(f'len of dataset_total: {len(dataset_total)}')
print(f'len of dataset_test: {len(dataset_test)}')
print(f'len(dataset_total) - len(dataset_test): {len(dataset_total) - len(dataset_test)}')
print(f'len(dataset_total) - len(dataset_test) - 60: {len(dataset_total) - len(dataset_test) - 60}')

#slice the dataset_total, get from the index 1198 to the end (about 80 rows)
inputs = dataset_total[ len(dataset_total) - len(dataset_test) - 60 :  ].values

print(f'len of inputs: {len(inputs)}')
print('----------------------------------------------')
print(f'first 10 rows of inputs: \n{inputs[0:10]}')

print('----------------------------------------------')
# -1 means infer this dimension from the length of the array, and 1 means we want to reshape it to have 1 column
#  so the input will be a 2D array, with n rows and 1 column, or n arrays of 1 element each,
#  as opposed to a 1D array with n elements
print(f'first 10 rows of inputs before reshape: \n{inputs[0:10]}')
print(f'inputs.shape: {inputs.shape}')
inputs = inputs.reshape(-1,1)
print(f'first 10 rows of inputs after reshape: \n{inputs[0:10]}')
print(f'inputs.shape: {inputs.shape}')


inputs = sc.transform(inputs)

print('----------------------------------------------')

x_test = []
for i in range(60, 80): #starting at 60 to offset the 60 timesteps slice going to 79
    # ranging from rows 0 to 59, selects 60 rows at a time, and select column 0
    x_test.append( inputs[ i - 60 : i, 0 ] )

x_test = np.array(x_test)
print(f'first 10 rows of x_test: \n{x_test[0:10]}')
x_test = np.reshape( x_test, (x_test.shape[0], x_test.shape[1], 1) ) #transforming to 3D array


print('----------------------------------------------')
print('Make a prediction')
predicted_stock_price = regressor.predict(x_test) #predicted stock price ranging from 0 to 1
print(f'first 10 rows of predicted_stock_price: \n{predicted_stock_price[0:10]}')
predicted_stock_price = sc.inverse_transform(predicted_stock_price) #transform the stock price back to actual values
print('----')
print(f'first 10 rows of predicted_stock_price: \n{predicted_stock_price[0:10]}')
print('----------------------------------------------')


print('Visualising the results')
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()

exit()
#--------------------------------


