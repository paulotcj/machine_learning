print('----------------------------------------------')
print('Thompson Sampling')

print('----------------------------------------------')
print('Importing the libraries')

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

print('----------------------------------------------')
print('Importing the dataset')

dataset = pd.read_csv('Ads_CTR_Optimisation.csv')


print('----------------------------------------------')
print('Implementing Thompson Sampling')

#----------------------------------------------
def thompson_sampling(dataset, sub_sample_size = None):
    import random
    N = dataset.shape[0] #number of rows (10_000) - for this example this is the number of users the ads were displayed to
    d = dataset.shape[1] #number of columns (10) - for this example this is the number of ads

    if sub_sample_size is not None: N = sub_sample_size #if we want to use a sub-sample of the dataset
    #---
    ads_selected = []
    numbers_of_rewards_1 = [0] * d # (N1i(n)) number of times the ad i got reward 1 up to round n
    numbers_of_rewards_0 = [0] * d # (N0i(n)) number of times the ad i got reward 0 up to round n
    #---
    total_reward = 0
    #--------------
    for n in range(0, N): #loop through each round (0 - 9_999 typically)
        ad = 0 #start considering ad 0 - but at each round we will select only 1 ad
        max_random_beta_distribution = 0 #considering we will select the max random value, we need to keep track of max per round
        
        #--------------
        for i in range(0, d): #loop through each ad (0 - 9)
            #--- 
            # The Beta distribution is a continuous probability distribution defined on the interval 
            #   [0,1]. It is parameterized by two positive shape parameters, typically denoted as 
            #   α and β, which determine the shape of the distribution. The Beta distribution is 
            #   particularly useful in Bayesian statistics and is often used as a prior distribution 
            #   for binomial proportions.
            
            random_beta_distribution = random.betavariate(
                numbers_of_rewards_1[i] + 1, 
                numbers_of_rewards_0[i] + 1
            )
            #---
            # now is this new random value greater than the max random value we have seen so far?
            if random_beta_distribution > max_random_beta_distribution:
                max_random_beta_distribution = random_beta_distribution
                ad = i #keep track of which ad was selected with the max beta distribution
            #---
        # end of for i in range(0, d):
        #--------------
        
        ads_selected.append(ad) #finished with all ads, by now we should know which ad was selected
        
        #get the reward for the selected ad - Note: we are considering that the ad not being clicked 
        #  reward 0, in a sense that the algorithm is not particularly interested wether the ad was
        #  clicked or not - so you can make your own decisions. For instance you might want to target
        #  the ads with fewer clicks so to not to spend money with them
        reward = dataset.values[n, ad] 
        
        if reward == 1:
            numbers_of_rewards_1[ad] = numbers_of_rewards_1[ad] + 1
        else:
            numbers_of_rewards_0[ad] = numbers_of_rewards_0[ad] + 1
            
        total_reward = total_reward + reward

    # end of for n in range(0, N):
    #--------------
    
    return ads_selected
#----------------------------------------------


print('----------------------------------------------')

# Note: The results will vary. Even more when we decrease the sample size. As it is a random algorithm
#  from tests, sample sizes 1250 and below might be quite unstable and produce the wrong results
#  however, with multiple runs, on average, the algorithm picked the right choice, even with a smaller
#  number than UCB.
# So the key take away from these tests is: 
#  1 - UCB is deterministic. Thompson Sampling is probabilistic.
#  2 - UCB will fail with smaller sample sizes. Thompson Sampling might still work on average runs
#  3 - As we increase the sample UCB, for being deterministic, will converge to the right choice
#        while Thompson being probalisitic, might point to the wrong choice, but on average, 
#        it will get it right


print('----------------------------------------------')
ads_selected = thompson_sampling(dataset)
print('Visualising the results - Histogram')
plt.hist(ads_selected)
plt.title('Histogram of ads selections - Sample size: 10_000')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()
print('----------------------------------------------')
ads_selected = thompson_sampling(dataset, sub_sample_size=5000)
print('Visualising the results - Histogram')
plt.hist(ads_selected)
plt.title('Histogram of ads selections - Sample size: 5000')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()
print('----------------------------------------------')
ads_selected = thompson_sampling(dataset, sub_sample_size=2500)
print('Visualising the results - Histogram')
plt.hist(ads_selected)
plt.title('Histogram of ads selections - Sample size: 2500')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()
print('----------------------------------------------')
ads_selected = thompson_sampling(dataset, sub_sample_size=1250)
print('Visualising the results - Histogram')
plt.hist(ads_selected)
plt.title('Histogram of ads selections - Sample size: 1250')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()
print('----------------------------------------------')
ads_selected = thompson_sampling(dataset, sub_sample_size=625)
print('Visualising the results - Histogram')
plt.hist(ads_selected)
plt.title('Histogram of ads selections - Sample size: 625')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()
print('----------------------------------------------')
ads_selected = thompson_sampling(dataset, sub_sample_size=500)
print('Visualising the results - Histogram')
plt.hist(ads_selected)
plt.title('Histogram of ads selections - Sample size: 500 - Same as the example')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()
print('----------------------------------------------')
ads_selected = thompson_sampling(dataset, sub_sample_size=312)
print('Visualising the results - Histogram')
plt.hist(ads_selected)
plt.title('Histogram of ads selections - Sample size: 312')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()