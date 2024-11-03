from random import seed
from random import randint
import numpy as np


#-------------------------------------------------------------------------
def normalize_values(value : np.array, n_numbers : int, largest : int):
    # In this working script we are generating random integer between 1 and 100, so the
    #  largest unique random value is 100, and since we could also be normalizing the sum
    #  of these numbers, the largest possible sum is 200, so imagine the examples:
    #    100 / 200 = 0.5, 1 / 200 = 0.005 , 30 / 200 = 0.15, and
    #    200 / 200 = 1
    return value.astype('float') / float(largest * n_numbers)
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
# invert normalization
def invert_normalization(value, n_numbers, largest):
    return round(value * float(largest * n_numbers))
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
# generate examples of random integers and their sum
def random_sum_pairs(n_examples, n_numbers, largest):
    x, y = list(), list()
    for i in range(n_examples):
        in_pattern = [randint(1,largest) for _ in range(n_numbers)] #random integers between 1 and the 'largest' value
        out_pattern = sum(in_pattern)
        x.append(in_pattern) # these numbers are meant to be added up in y
        y.append(out_pattern) #this is the result of the sum in x)
    # format as NumPy arrays
    x,y = np.array(x), np.array(y)
    # normalize
    x = normalize_values(x, n_numbers, largest)
    y = normalize_values(y, n_numbers, largest)
    return x, y
#-------------------------------------------------------------------------

n_examples, n_numbers , largest = 10_000, 2, 100
n_epoch, n_batch = 100, 1

x, y = random_sum_pairs(n_examples=n_examples, n_numbers=n_numbers, largest=largest)
print(f'x first 5 rows: {x[:5]}')
print('--------')
print(f'y first 5 rows: {y[:5]}')

print('----------------------------------------------')
print('Defining the LSTM model')

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

import os
from keras.models import load_model





# time to configure the LSTM model, the input layer will receive 1 input feature with 
#  2 time steps (the 2 numbers to be added up)


# create LSTM
model = Sequential()

model.add(LSTM(units = 6, input_shape=(n_numbers, 1), return_sequences=True))
model.add(LSTM(units = 6)) #no input_shape needed, its inferred from the previous layer, and return_sequences = False, meaning this layer will only return the last output
model.add(Dense(units = 1))
model.compile(loss='mean_squared_error', optimizer='adam')


print('----------------------------------------------')
print('Training or loadint the LSTM model')
# train LSTM

x, y = None, None

model_path = 'seq2seq_model.h5'

if os.path.exists(model_path):
    print('Loading the existing model...')
    model = load_model(model_path)
else:
    print('Training the model...')
    # model.fit(x_train, y_train, epochs=100, batch_size=32)

    
    for _ in range(n_epoch):
        x, y = random_sum_pairs(n_examples=n_examples, n_numbers=n_numbers, largest=largest)
        # print(f'x first 5 rows: {x[:5]}')
        # print('--------')
        # print(f'y first 5 rows: {y[:5]}')
        x = x.reshape(n_examples, n_numbers, 1) #set the appropriate dimension, usually 100, 2, 1
        model.fit(x, y, epochs=1, batch_size=n_batch, verbose=2)

    print('Saving the model...')
    model.save(model_path)

print('----------------------------------------------')
print('Evaluate on some new patterns')

x , y = random_sum_pairs(n_examples = n_examples, n_numbers = n_numbers, largest = largest)
x = x.reshape(n_examples, n_numbers, 1) #set the appropriate dimension, usually 100, 2, 1

result = model.predict(x, batch_size = n_batch, verbose = 2)


print(f'result first 5 rows: {result[:5]}')
print('--------')
print(f'y first 5 rows: {y[:5]}')

print('----------------------------------------------')
print('calculate error')


from sklearn.metrics import mean_squared_error
from math import sqrt

expected = [ 
        invert_normalization(value = y_i , n_numbers = n_numbers, largest = largest) 
        for y_i in y 
    ]

predicted = [ 
        invert_normalization(value = result_i, n_numbers = n_numbers, largest = largest) 
        for result_i in result[:, 0]  #all rows, first column , since this is a 2d array of n elements with 1 array of 1 element as column
    ]



root_mean_squared_error = sqrt( mean_squared_error(expected, predicted) )



print(f'root mean squared error: {root_mean_squared_error}')

print('----------------------------------------------')
print('Show predictions')

print('    Expected  |  Predicted  |  Error   ')
print('    ___________________________________')
#          12345678  |  12345678   |  12345678  
for i in range(len(expected)):
    error = expected[i] - predicted[i]

    print(f'    {expected[i]:<8}  |  {predicted[i]:<8}   |  {error:<8}')


    

