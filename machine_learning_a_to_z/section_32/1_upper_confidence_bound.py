print('----------------------------------------------')
print('Upper Confidence Bound (UCB)')

print('----------------------------------------------')
print('Importing the libraries')

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

print('----------------------------------------------')
print('Importing the dataset')
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

print('dataset (sample 5 first rows):')
print(dataset[0:5])


print('----------------------------------------------')
print('Implementing UCB')
#----------------------------------------------
def UCB(dataset, sub_sample_size = None):
    import math
    #---
    N = dataset.shape[0] #number of rows (10_000) - for this example this is the number of users the ads were displayed to
    d = dataset.shape[1] #number of columns (10) - for this example this is the number of ads
    
    if sub_sample_size is not None: N = sub_sample_size #if we want to use a sub-sample of the dataset
    #---
    ads_selected = [] #list of ads selected at each round
    #---
    numbers_of_selections = [0] * d # Ni(n)
    sums_of_rewards = [0] * d       # Ri(n)
    #---
    total_reward = 0
    #--------------
    for n in range(0, N): #loop through each round (0 - 9_999)
        max_upper_bound = 0 #keep track of the maximum upper bound - start at 0
        # we should select the ad that has the maximum upper confidence bound
        ad = 0
        
        #--------------
        for i in range(0, d): #loop through each ad (0 - 9)
            #---
            # we need to check if the ad was selected before to avoid a division by zero
            if (numbers_of_selections[i] == 0): # if the was was not selected - then just set the UCB to infinity
                upper_bound = 1e400 #we might need to change this to infinity
            else:
                
                average_reward = sums_of_rewards[i] / numbers_of_selections[i] # /ri(n) - average reward
                
                #   we can't do log(0) so we add 1 to the number of rounds - on the plus side this reflect more accurately the number of rounds
                delta_i = math.sqrt(3/2 * math.log(n + 1) / numbers_of_selections[i]) # delta_i(n) - confidence interval delta
                
                #note we are not using the confidence interval here: [/ri(n) - delta_i(n), /ri(n) + delta_i(n)]
                
                upper_bound = average_reward + delta_i # upper confidence bound UCBi
                
            #---

            # in the first round all ads will have the same (extremely large) upper bound
            #  then we start to reduce that
            # but also here, we are trying to select the AD that was clicked and has the highest upper bound
            if upper_bound > max_upper_bound:
                max_upper_bound = upper_bound
                ad = i
                
            #---
        # end of - for i in range(0, d):
        #--------------
        
        # so we finished with the the round, now we need to update the lists
        ads_selected.append(ad) #just to keep track of which ad was selected (higher UCB) at each round
        numbers_of_selections[ad] = numbers_of_selections[ad] + 1
        reward = dataset.values[n, ad]
        sums_of_rewards[ad] = sums_of_rewards[ad] + reward
        total_reward = total_reward + reward
        
    # end of - for n in range(0, N):
    #--------------
    
    return ads_selected
#----------------------------------------------

print('----------------------------------------------')
print('Visualising the results')

ads_selected = UCB(dataset)
plt.hist(ads_selected)
plt.title('Histogram of ads selections - Sample size: 10_000')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()
#---------------
ads_selected = UCB(dataset, sub_sample_size = 5000)
plt.hist(ads_selected)
plt.title('Histogram of ads selections - Sample size: 5000')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()
#---------------
ads_selected = UCB(dataset, sub_sample_size = 2500)
plt.hist(ads_selected)
plt.title('Histogram of ads selections - Sample size: 2500')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()
#---------------
ads_selected = UCB(dataset, sub_sample_size = 1250)
plt.hist(ads_selected)
plt.title('Histogram of ads selections - Sample size: 1250')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()
#---------------
ads_selected = UCB(dataset, sub_sample_size = 625)
plt.hist(ads_selected)
plt.title('Histogram of ads selections - Sample size: 625')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()
#---------------
ads_selected = UCB(dataset, sub_sample_size = 500)
plt.hist(ads_selected)
plt.title('Histogram of ads selections - Sample size: 500 - Same as the example')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()
#---------------
ads_selected = UCB(dataset, sub_sample_size = 312)
plt.hist(ads_selected)
plt.title('Histogram of ads selections - Sample size: 312')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()
