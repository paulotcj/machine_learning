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


transactions = [ [ str(col) for col in row] for row in dataset.values ]
# transactions = []
# for i in range(0, 7501):
#   transactions.append([str(dataset.values[i,j]) for j in range(0, 20)])

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
rules = apriori(transactions = transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2, max_length = 2)



print('----------------------------------------------')
print('Visualising the results')

print('----------------------------------------------')
print('Displaying the first results coming directly from the output of the apriori function')

# print(rules)
results = list(rules)
for i in results:
    print(i)
    print()
    


print('----------------------------------------------')
print('Putting the results well organised into a Pandas DataFrame')

#----------------------------------------------
def local_format_results(results):
    lhs         = [tuple(result[2][0][0])[0] for result in results]
    rhs         = [tuple(result[2][0][1])[0] for result in results]
    supports    = [result[1] for result in results]
    confidences = [result[2][0][2] for result in results]
    lifts       = [result[2][0][3] for result in results]
    return list(zip(lhs, rhs, supports, confidences, lifts))
#----------------------------------------------

results_in_dataframe = pd.DataFrame(local_format_results(results), columns = ['Left Hand Side', 
            'Right Hand Side', 'Support', 'Confidence', 'Lift'])


print('----------------------------------------------')
print('Displaying the results non sorted')


print('resultsinDataFrame:\n')
print(results_in_dataframe)
print('\nLegend: Left Hand Side: product bought.')
print('Right Hand Side: product bought together with Right Hand Side')
print('Support: number of times the product was bought in a week / total number of transactions (7501)')
print('Confidence: number of times the products were bought together / number the Left Hand Side was bought')
print('Lift: Lift(A -> B) = Support(A ∪ B) / (Support(A) * Support(B)). The higher the lift, the higher the value the more likely the rule is true')



print('----------------------------------------------')
print('Displaying the results sorted by descending lifts')

print(
    results_in_dataframe.nlargest(n = 10, columns = 'Lift') # sort by descending lifts, get only the 10 first rows
)

