from random import seed
from random import randint
from numpy import array
from math import ceil
from math import log10
from math import sqrt
from numpy import argmax
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.layers import RepeatVector

# generate lists of random integers and their sum
def random_sum_pairs(n_examples, n_numbers, largest):
    X, y = list(), list()
    for i in range(n_examples):
        in_pattern = [randint(1,largest) for _ in range(n_numbers)]
        out_pattern = sum(in_pattern)
        X.append(in_pattern)
        y.append(out_pattern)
    return X, y

# convert data to strings
def to_string(X, y, n_numbers, largest):
    max_length = n_numbers * ceil(log10(largest+1)) + n_numbers - 1
    Xstr = list()
    for pattern in X:
        strp = '+'.join([str(n) for n in pattern])
        strp = ''.join([' ' for _ in range(max_length-len(strp))]) + strp
        Xstr.append(strp)
    max_length = ceil(log10(n_numbers * (largest+1)))
    ystr = list()
    for pattern in y:
        strp = str(pattern)
        strp = ''.join([' ' for _ in range(max_length-len(strp))]) + strp
        ystr.append(strp)
    return Xstr, ystr

# integer encode strings
def integer_encode(X, y, alphabet):
    char_to_int = dict((c, i) for i, c in enumerate(alphabet))
    Xenc = list()
    for pattern in X:
        integer_encoded = [char_to_int[char] for char in pattern]
        Xenc.append(integer_encoded)
    yenc = list()
    for pattern in y:
        integer_encoded = [char_to_int[char] for char in pattern]
        yenc.append(integer_encoded)
    return Xenc, yenc

# one hot encode
def one_hot_encode(X, y, max_int):
    Xenc = list()
    for seq in X:
        pattern = list()
        for index in seq:
            vector = [0 for _ in range(max_int)]
            vector[index] = 1
            pattern.append(vector)
        Xenc.append(pattern)
    yenc = list()
    for seq in y:
        pattern = list()
        for index in seq:
            vector = [0 for _ in range(max_int)]
            vector[index] = 1
            pattern.append(vector)
        yenc.append(pattern)
    return Xenc, yenc

# generate an encoded dataset
def generate_data(n_samples, n_numbers, largest, alphabet):
    # generate pairs
    X, y = random_sum_pairs(n_samples, n_numbers, largest)
    # convert to strings
    X, y = to_string(X, y, n_numbers, largest)
    # integer encode
    X, y = integer_encode(X, y, alphabet)
    # one hot encode
    X, y = one_hot_encode(X, y, len(alphabet))
    # return as numpy arrays
    X, y = array(X), array(y)
    return X, y

# invert encoding
def invert(seq, alphabet):
    int_to_char = dict((i, c) for i, c in enumerate(alphabet))
    strings = list()
    for pattern in seq:
        string = int_to_char[argmax(pattern)]
        strings.append(string)
    return ''.join(strings)


print('----------------------------------------------')
print('define dataset')
seed(1)
n_samples = 1000
n_numbers = 2
largest = 10
alphabet = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', ' ']
n_chars = len(alphabet)
n_in_seq_length = n_numbers * ceil(log10(largest+1)) + n_numbers - 1
n_out_seq_length = ceil(log10(n_numbers * (largest+1)))

print('----------------------------------------------')
print('define LSTM configuration')
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
print('train LSTM')

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
print('evaluate on some new patterns')

n_samples = 1000

x, y = generate_data(n_samples, n_numbers, largest, alphabet)
result = model.predict(x, batch_size=n_batch, verbose=0)
# calculate error
expected = [invert(i, alphabet) for i in y]
predicted = [invert(i, alphabet) for i in result]

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