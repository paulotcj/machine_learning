
from random import seed
from random import randint
import numpy as np
# from numpy import array
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.layers import LSTM
from math import sqrt
from sklearn.metrics import mean_squared_error

seed(1)
x, y = list(), list()
for i in range(100):
	in_pattern = [randint(1,100) for _ in range(2)] #random integers between 1 and 100
	out_pattern = sum(in_pattern)
	print(f'The sum of these numbers: {in_pattern} is {out_pattern}')
	x.append(in_pattern)
	y.append(out_pattern)
	
print(f'x first 5 rows (these numbers are meant to be added up in y\n: {x[:5]}')

print('--------')
print(f'y first 5 rows (this is the result of the sum in x)\n: {y[:5]}')

print('----------------------------------------------')
# format as NumPy arrays
x,y = np.array(x), np.array(y)
print(f'x first 5 rows: {x[:5]}')
print('--------')
print(f'y first 5 rows: {y[:5]}')
print('--------')
print(f'Normalize x and y')
# normalize
# suppose: 100 / 200 = 0.5, 1 / 200 = 0.005 , 30 / 200 = 0.15
x = x.astype('float') / float(100 * 2)
y = y.astype('float') / float(100 * 2) # and since the max val here is 100 + 100 = 200, if we do 200/200

print(f'x first 5 rows: {x[:5]}')
print('--------')
print(f'y first 5 rows: {y[:5]}')


