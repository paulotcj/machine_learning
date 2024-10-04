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


# Parameter split_percent defines the ratio of training examples
def get_train_test(url, split_percent=0.8):
    df = read_csv(url, usecols=[1], engine='python')
    data = np.array(df.values.astype('float32'))

    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(data).flatten()

    n = len(data)
    # Point for splitting data into train and test
    split = int(n*split_percent)

    train_data = data[range(split)]
    test_data = data[split:]

    return train_data, test_data, data

sunspots_url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/monthly-sunspots.csv'
train_data, test_data, data = get_train_test(sunspots_url)

print('-----')
print(f'train_data first 5 rows : {train_data[0:5]}')
print('-----')
print(f'test_data first 5 rows : {test_data[0:5]}')
print('-----')
print(f'data first 5 rows : {data[0:5]}')
print('-----')