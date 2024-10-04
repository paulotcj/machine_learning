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

print(f'Xt: not available at this stage')
print(f'Ht: not available at this stage\n')

print(f'Wx:{wx}')
print(f'Wh:{wh}')
print(f'Bh:{bh}\n')

print(f'Wy:{wy}')
print(f'By:{by}\n')

print(f'Yt: not available at this stage\n')
print('----------------------------------------------')

# print(f'wx = {wx}, wh = {wh}, bh = {bh}, wy = {wy}, by = {by}')

exit()