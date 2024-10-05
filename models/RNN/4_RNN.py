from pandas import read_csv
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import math
import matplotlib.pyplot as plt

                        
#-------------------------------------------------------------------------
def create_RNN(hidden_units, dense_units, input_shape, activation):
    model = Sequential()
    model.add(SimpleRNN(hidden_units, input_shape=input_shape, 
                        activation=activation[0]))
    model.add(Dense(units=dense_units, activation=activation[1]))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model
#-------------------------------------------------------------------------


#input_shape = (time_steps, features)
demo_model = create_RNN(hidden_units=2 , dense_units=1, input_shape = (3,1),
                        activation = ['linear', 'linear'])
print('----------------------------------------------')
# More info at: https://machinelearningmastery.com/an-introduction-to-recurrent-neural-networks-and-the-math-that-powers-them/
print('RNN definitions \n  ***** For a network of M hidden units *****:')
print('  Xt = input at time t')
print('  Ht = hidden vector state at time t\n')

print('  Wx = weights for input          - array of len M')
print('  Wh = weights for hidden state   - array of len MxM - Note: Each hidden unit in the current time step is connected to each hidden unit in the previous time step')
print('  Bh = bias for the hidden state  - array of len M \n')

print('  Wy = weights for the hidden output layer  - array o len M')
print('  By = bias for the output layer            - array of len M\n')

print('  Yt = the final output of the network at time t \n')
print('----------------------------------------------')

wx = demo_model.get_weights()[0]
wh = demo_model.get_weights()[1]
bh = demo_model.get_weights()[2]
wy = demo_model.get_weights()[3]
by = demo_model.get_weights()[4]
print(f'The data below is randomly initialized\n')
print(f'Xt: not available at this stage')
print(f'Ht: not available at this stage\n')

print(f'Wx:{wx}')
print(f'Wh:{wh}')
print(f'Bh:{bh}\n')

print(f'Wy:{wy}')
print(f'By:{by}\n')

print(f'Yt: not available at this stage\n')
print('----------------------------------------------')
print(f'Initializing the input to the network and making a prediction')
x = np.array([1, 2, 3])
# Reshape the input to the required sample_size x time_steps x features 
x_input = np.reshape(x , (1, 3, 1))
print(f'x_input shape: {x_input.shape}')
print(f'x_input: {x_input}')

y_pred_model = demo_model.predict(x_input)

print('----------------------------------------------')
print(f'Manual computation of the output')
print(f'  We will compute 3 hidden states, so  we will have the initial hidden state h0, plus h1, h2, h3')
print(f'  Note: We will be using x (1D array on len 3) in the calculations, not x_input\n')
m = 2
h0 = np.zeros(m)
h1 = np.dot(x[0], wx) + np.dot(h0,wh) + bh
h2 = np.dot(x[1], wx) + np.dot(h1,wh) + bh
h3 = np.dot(x[2], wx) + np.dot(h2,wh) + bh
o3 = np.dot(h3, wy) + by

print(f'  size of hidden units (m):{m}')
print(f'  h0:{h0} - h1:{h1} -  h2:{h2} -  h3:{h3}')

print("  Prediction from network ", y_pred_model)
print("  Prediction from our computation ", o3)

print('----------------------------------------------')


#-------------------------------------------------------------------------
# Parameter split_percent defines the ratio of training examples
def get_train_test_local(file = 'monthly-sunspots.csv', split_percent=0.8):
    data_frame = read_csv(f'./{file}', usecols=[1], engine='python')

    data = np.array(data_frame.values.astype('float32'))

    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(data).flatten()

    len_data = len(data)
    # Point for splitting data into train and test
    split = int(len_data * split_percent)

    train_data = data[range(split)] #range return like this will slice the array
    test_data = data[split:]

    return train_data, test_data, data
#-------------------------------------------------------------------------

train_data, test_data, data = get_train_test_local()

print('-----')
print(f'len of train_data : {len(train_data)}')
print(f'train_data first 5 rows : {train_data[0:5]}')
print('-----')
print(f'len of test_data : {len(test_data)}')
print(f'test_data first 5 rows : {test_data[0:5]}')
print('-----')
print(f'len of data : {len(data)}')
print(f'data first 5 rows : {data[0:5]}')
print('-----')

print('----------------------------------------------')

# The idea in this section is to slice the data into non-overlapping chunks of time_steps
# suppose the data = [0, 10, 20, 30, 40, 50, 60, 70] - and we want a time_step of 2 and to predict
# the 3rd value in the sequence. So at:
# t [0, 1] -> x = [0, 10] -> predict t2 [20]
# t [2, 3] -> x = [20, 30] -> predict t4 [40]
# t [4, 5] -> x = [40, 50] -> predict t6 [60]
# and discard 70 because even though we could pair 60 and 70 we don't have a data to predict 
# and compare the results.

#-------------------------------------------------------------------------
# Prepare the input X and target Y
def get_xy(param_data, time_steps = 12):
    
    #Generate an array of indices, start at time_steps, increment by time_steps, and end at len(param_data)
    # so in an example with time_steps = 12: 12, 24, 36, 48, ... 
    y_indexes = np.arange(time_steps, len(param_data), time_steps)

    y_data = param_data[y_indexes]

    rows_y = len(y_data)

    #-----------
    # Prepare X

    #we are trying to predict the result of every th time_step, so we know y_data is the results we are trying
    # to predict, therefore the relevant data in x should be chunks of time_steps (default 12)
    
    x_data = param_data[range(time_steps*rows_y)]


    #In the example we would have x_data with 2244 elements, y_rows with 187, and time_steps = 12 . 
    # So rows_y * time_steps * 1 = 187 * 1 = 187 * 12 * 1 = 2244
    # We have a 3D array, with 187 samples, with 12 rows, and 1 column
    # To quote the source material: "The input array should be shaped as: total_samples x time_steps 
    #  x features."
    x_data = np.reshape(x_data, (rows_y, time_steps, 1))   
    return x_data, y_data
#-------------------------------------------------------------------------

time_steps = 12
train_x, train_y = get_xy( train_data, time_steps )
test_x , test_y  = get_xy( test_data , time_steps )


print(f'len of train_x : {len(train_x)} - shape of train_x: {train_x.shape}')
print(f'len of train_y : {len(train_y)} - shape of train_y: {train_y.shape}')
print('-----')
print(f'len of test_x : {len(test_x)} - shape of train_y: {test_x.shape}')
print(f'len of test_y : {len(test_y)} - shape of train_y: {test_y.shape}')
print('-----')
# print(f'train_x first 5 rows : {train_x[0:5]}')
# print(f'train_y first 5 rows : {train_y[0:5]}')
# print(f'test_x first 5 rows : {test_x[0:5]}')
# print(f'test_y first 5 rows : {test_y[0:5]}')



print('----------------------------------------------')
print('Create the RNN model and train it')
model = create_RNN(hidden_units=3, dense_units=1, input_shape=(time_steps,1), 
                   activation=['tanh', 'tanh'])

# batch_size=1: This parameter defines the number of samples that will be propagated through 
#  the model before updating the model's weights. A batch size of 1 means that the model's 
#  weights will be updated after each sample, which is also known as online learning.
model.fit(train_x, train_y, epochs=20, batch_size=1, verbose=2)

print('----------------------------------------------')
