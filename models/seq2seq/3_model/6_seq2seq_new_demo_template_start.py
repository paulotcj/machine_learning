from random import seed
from random import randint
import numpy as np
import math

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



seed(1)
# n_samples = 1
n_samples = 10_000
n_numbers = 2
# largest = 10
largest = 999
# generate pairs



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


