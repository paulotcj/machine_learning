from random import seed
from random import randint
import numpy as np
# from numpy import array
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.layers import LSTM
# from math import sqrt
# from sklearn.metrics import mean_squared_error

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

x , y = random_sum_pairs(n_examples=100, n_numbers=2, largest=100)
print(f'x first 5 rows: {x[:5]}')
print('--------')
print(f'y first 5 rows: {y[:5]}')
print('--------')
exit()


