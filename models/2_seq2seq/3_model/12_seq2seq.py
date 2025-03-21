from random import seed
from random import randint
import numpy as np
import math

# based on: https://machinelearningmastery.com/learn-add-numbers-seq2seq-recurrent-neural-networks/

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
# generate lists of random integers and their sum
def random_sum_pairs(n_examples, n_numbers, largest):
    x, y = list(), list()
    for i in range(n_examples):
        in_pattern = [randint(1,largest) for _ in range(n_numbers)] #random integers between 1 and the 'largest' value
        out_pattern = sum(in_pattern)
        x.append(in_pattern) #these numbers are meant to be added up in y
        y.append(out_pattern) #this is the result of the sum in x)

    return x, y
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
# convert data to strings
def to_string(x, y, n_numbers, largest):
    # max_length = n_numbers * math.ceil(math.log10(largest+1)) + n_numbers - 1
    max_length = len(str(largest))*n_numbers + (n_numbers-1)

    x_str = list()
    for pattern in x:
        str_pat = '+'.join([str(n) for n in pattern])
        str_pat = ''.join([' ' for _ in range(max_length-len(str_pat))]) + str_pat
        x_str.append(str_pat)
        


    # max_length = math.ceil(math.log10(n_numbers * (largest+1)))

    max_length = math.ceil(math.log10(largest * n_numbers))
    ystr = list()



    for pattern in y:
        str_pat = str(pattern)
        str_pat = ''.join([' ' for _ in range(max_length-len(str_pat))]) + str_pat
        ystr.append(str_pat)

    #check routine
    for i in range(len(x)):
        sum_result = sum([ x_i for x_i in x[i] ])

        if sum_result != y[i]:
            print(f'Error: {x[i]} = {sum_result} != {y[i]}')
            exit()
    return x_str, ystr
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
# integer encode strings
def integer_encode(x, y, alphabet):

    # creates a dict with the index being the char and the value being the number
    #  so we will have: '0':0, '1':1, ...
    char_to_int = dict((v, k) for k, v in enumerate(alphabet))

    #lets encode the x and y strings. Please note that symbols like '+' and ' ' are also encoded
    #  so '+' will likely be represented by the number 10 and ' ' by 11
    x_enc = []
    for pattern in x:
        integer_encoded = [char_to_int[char] for char in pattern]
        x_enc.append(integer_encoded)

    y_enc = []
    for pattern in y:
        integer_encoded = [char_to_int[char] for char in pattern]
        y_enc.append(integer_encoded)

    return x_enc, y_enc
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
def one_hot_encode(x, y, alphabet_len):
    
    #----------
    x_enc = []
    for sequence in x:
        pattern = []
        
        for e in sequence:
            vector = [0] * alphabet_len
            vector[e] = 1
            pattern.append(vector)

        x_enc.append(pattern)
    #----------
    y_enc = []
    for sequence in y:
        pattern = []
        vector = [0] * alphabet_len
        for e in sequence:
            vector = [0] * alphabet_len
            vector[e] = 1
            pattern.append(vector)

        y_enc.append(pattern)
    #----------
    return x_enc, y_enc
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
# generate an encoded dataset
def generate_data(n_samples, n_numbers, largest, alphabet):
    # generate pairs
    x, y = random_sum_pairs(n_examples=n_samples, n_numbers=n_numbers, largest=largest)
    # convert to strings
    x, y = to_string(x, y, n_numbers, largest)
    # integer encode
    x, y = integer_encode(x, y, alphabet)
    # one hot encode
    x, y = one_hot_encode(x, y, len(alphabet))
    # return as numpy arrays
    x, y = np.array(x), np.array(y)
    return x, y
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
# invert encoding
def invert_one_hot_encode(seq, alphabet):

    #alphabet is a list of chars, so for '1' we will have 1, for '2' we will have 2, and so on
    # typically '+' will be 10 and ' ' will be 11
    # so in the case below we will have 1 : '1', 2 : '2', ...
    int_to_char = dict((k, v) for k, v in enumerate(alphabet))
    strings = []
    for pattern in seq:
        #argmax returns the index of the max value - so what we are doing is to get the index of the 1 in the one hot encoded vector
        # and then we are getting the char that represents that index
        string = int_to_char[np.argmax(pattern)] 
        strings.append(string)
    return ''.join(strings)
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
def start_test1():
    seed(1)
    # n_samples = 1
    n_samples = 10_000
    n_numbers = 2
    # largest = 10
    largest = 999
    # generate pairs


    # x, y = random_sum_pairs(n_samples, n_numbers, largest)

    x, y = random_sum_pairs(n_examples=n_samples, n_numbers=n_numbers, largest=largest)

    print(f'x len: {len(x)}')
    print(f'x first 5 rows: {x[:5]}')
    print('--------')
    print(f'y len: {len(y)}')
    print(f'y first 5 rows: {y[:5]}')
    print('----------------------------------------------')

    x, y = to_string(x, y, n_numbers, largest)
    print(f'x len: {len(x)}')
    print(f'x first 5 rows: {x[:5]}')
    print('--------')
    print(f'y len: {len(y)}')
    print(f'y first 5 rows: {y[:5]}')
    print('----------------------------------------------')



    # integer encode
    alphabet = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', ' ']

    print(f'Encoding alphabet')
    x, y = integer_encode(x, y, alphabet)

    print(f'x len: {len(x)}')
    print(f'x first 5 rows: \n{x[:5]}')
    print('--------')
    print(f'y len: {len(y)}')
    print(f'y first 5 rows: \n{y[:5]}')
    print('----------------------------------------------')

    print(f'One hot encode')
    x_backup = x
    y_backup = y
    x, y = one_hot_encode(x, y, len(alphabet))
    print(f'x len: {len(x)}')
    print(f'x first 5 rows: \n{x[:5]}')
    print('--------')
    print(f'y len: {len(y)}')
    print(f'y first 5 rows: \n{y[:5]}')
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
def start_test2_comments_may_be_relevant():
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import LSTM
    from keras.layers import TimeDistributed
    from keras.layers import RepeatVector


    seed(1)
    n_samples = 1
    # n_samples = 10_000

    n_numbers = 2

    largest = 10
    # largest = 999

    #            1    2    3    4    5    6    7    8    9    10   11   12
    alphabet = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', ' ']

    n_batch = 10
    n_epoch = 30

    model = Sequential()

    # for input shape: 2 digits integer + 1 signal + 2 digits integer = 5, then the alphabet size is (typically) 12
    model.add(LSTM(100, input_shape=(5, 12)))


    model.add(RepeatVector(2))


    # The first LSTM layer is defined with input_shape=(5, 11), which means it expects input 
    #  sequences of length 5 (timesteps) and each input at every timestep has 11 features.
    #
    # The first LSTM layer has units=100 but does not have return_sequences=True, meaning it 
    #  will return only the last hidden state (a vector of size 100) after processing the entire 
    #  sequence. This output will have the shape (batch_size, 100).
    #
    # The second LSTM layer, however, has return_sequences=True, which means IT EXPECTS TO RECEIVE A 
    #  SEQUENCE, not just a single hidden state. At minimum then (batch_size, sequence, features)
    #  or considering the example (batch_size , 2 , 11) or most likely (None, 2, 11)
    # But since the first LSTM layer returns only the 
    #  last hidden state (shape (batch_size, 100)), the input to the second LSTM will not be a 
    #  sequence, which will cause a mismatch in input shape and an error.
    model.add(LSTM(50, return_sequences=True))

    # Dense without TimeDistributed computes per Batch, TimeDistributed with Dense computes per Timestep
    # "In keras - while building a sequential model - usually the second dimension (one after sample dimension)
    #  - is related to a time dimension. This means that if for example, your data is 5-dim with 
    # (sample, time, width, length, channel) you could apply a convolutional layer using TimeDistributed 
    # (which is applicable to 4-dim with (sample, width, length, channel))" - https://stackoverflow.com/questions/47305618/what-is-the-role-of-timedistributed-layer-in-keras
    model.add(TimeDistributed(Dense(12, activation='softmax')))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print('----------------------------------------------')
    print(model.summary())

    print('----------------------------------------------')

    # evaluate on some new patterns
    x, y = generate_data(n_samples = n_samples, n_numbers = n_numbers, largest = largest, alphabet = alphabet)

    print(f'x shape: {x.shape}')
    print(f'y shape: {y.shape}')

    result = model.predict(x, batch_size=n_batch, verbose=0)

    # calculate error
    expected = [invert_one_hot_encode(x, alphabet) for x in y]
    predicted = [invert_one_hot_encode(x, alphabet) for x in result]


    # show some examples
    for i in range(20):
        print('Expected=%s, Predicted=%s' % (expected[i], predicted[i]))
#-------------------------------------------------------------------------
print('----------------------------------------------')
print('Defining parameters')
seed(1)
n_samples = 300_000
n_numbers = 2
largest = 999
alphabet = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', ' ']
n_chars = len(alphabet)
n_in_seq_length = n_numbers * math.ceil(math.log10(largest+1)) + n_numbers - 1
n_out_seq_length = math.ceil(math.log10(n_numbers * (largest+1)))


print('----------------------------------------------')
print('Defining model (LSTM)')
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.layers import RepeatVector
# define LSTM configuration
n_batch = 10
n_epoch = 30
# create LSTM
model = Sequential()
model.add(LSTM(100, input_shape=(n_in_seq_length, n_chars)))

model.add(RepeatVector(n_out_seq_length))
model.add(LSTM(50, return_sequences=True))

model.add(TimeDistributed(Dense(n_chars, activation='softmax')))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

print('----------------------------------------------')
print('Training the LSTM model')
# train LSTM
# for i in range(n_epoch):
#     x, y = generate_data(n_samples = n_samples, n_numbers = n_numbers, largest = largest, alphabet = alphabet)
#     model.fit(x, y, epochs=1, batch_size=n_batch)

import os
from keras.models import load_model

model_path = 'seq2seq_model.h5'

if os.path.exists(model_path):
    print('Loading the existing model...')
    model = load_model(model_path)
else:
    print('Training the model...')

    for i in range(n_epoch):
        x, y = generate_data(n_samples, n_numbers, largest, alphabet)
        model.fit(x, y, epochs=1, batch_size=n_batch)


    print('Saving the model...')
    model.save(model_path)

print('----------------------------------------------')
print('Check the results')

x, y = generate_data(n_samples = n_samples, n_numbers = n_numbers, largest = largest, 
                     alphabet = alphabet )
result = model.predict(x, batch_size = n_batch, verbose = 2 )

# calculate error
expected = [ invert_one_hot_encode(i, alphabet) for i in y ]
predicted = [ invert_one_hot_encode(i, alphabet) for i in result ]

wrong_count = 0
wrongs_array = []
# show some examples
print('    Expected  |  Predicted  |  Wrong?')
#          12345678  |  12345678   |  True
print('    __________________________________')
for k,v in enumerate(expected):
    if expected[k] != predicted[k]:
        wrong_count += 1
        wrong = "True"
        wrongs_array.append([expected[k], predicted[k]])
    else:
        wrong = "."

    print(f'    {expected[k]:<8}  |  {predicted[k]:<8}   |  {wrong}')
print('----------------------------------------------')
print('Wrongs list:')
for n  in wrongs_array:
    print(f'Expected: {n[0]} - Predicted: {n[1]}')
print('----------------------------------------------')
print(f'Total samples: {len(expected)} - Wrong samples: {wrong_count} - Accuracy: {100 - (wrong_count/len(expected)*100):.2f}%')