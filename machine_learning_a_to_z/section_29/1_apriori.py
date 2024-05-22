print('----------------------------------------------')
print('Apriori')

print('----------------------------------------------')
print('Run the following command in the terminal to install the apyori package: pip install apyori')

print('----------------------------------------------')
print('Importing the libraries')

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd




print('----------------------------------------------')
print('Data Preprocessing')

dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)


transactions = [ [col for col in row] for row in dataset.values ]

print('transactions (sample 5 first rows):')
print(transactions[0:5])


print('----------------------------------------------')
print('Training the Apriori model on the dataset')


from apyori import apriori
# min support -> the rationale is we want to find products that are bought at least 3 times a day, and this record
#   if from 1 week, with 7501 transactions, so: 3 * 7 / 7501 = 0.0027... so we use 0.003
# min confidence -> we want to find rules that are correct at least 20% of the time, so we use 0.2
# min lift -> we want to find rules that are at least 3 times more likely to be true than false, so we use 3
# min_length -> we want to find rules with at least 2 products
# max_length -> we want to find rules with at most 2 products
rules = apriori(transactions = transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_lenght = 2, max_length = 2)

print(rules)
exit()

print('----------------------------------------------')
print('Visualising the results')

print('----------------------------------------------')
print('Displaying the first results coming directly from the output of the apriori function')

results = list(rules)
# print(results)
exit()

print('----------------------------------------------')
print('Putting the results well organised into a Pandas DataFrame')
def inspect(results):
    lhs         = [tuple(result[2][0][0])[0] for result in results]
    rhs         = [tuple(result[2][0][1])[0] for result in results]
    supports    = [result[1] for result in results]
    confidences = [result[2][0][2] for result in results]
    lifts       = [result[2][0][3] for result in results]
    return list(zip(lhs, rhs, supports, confidences, lifts))
resultsinDataFrame = pd.DataFrame(inspect(results), columns = ['Left Hand Side', 'Right Hand Side', 'Support', 'Confidence', 'Lift'])

print('----------------------------------------------')
print('Displaying the results non sorted')
resultsinDataFrame

print('----------------------------------------------')
print('Displaying the results sorted by descending lifts')
resultsinDataFrame.nlargest(n = 10, columns = 'Lift')