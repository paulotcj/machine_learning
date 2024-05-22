print('----------------------------------------------')
print('K-Means Clustering')

print('----------------------------------------------')
print('Import the libraries')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

print('----------------------------------------------')
print('Import the dataset')

dataset = pd.read_csv('Mall_Customers.csv')
x = dataset.iloc[:, [3, 4]].values #import all rows, but only columns 3 and 4
# this is clustering, so we don't need to split the dataset into training and test sets
# also, because we plan to show this result in a 2D graph, we can only use 2 columns (2 dimensional plot)

print('x (sample 5 first rows):')
print(x[0:5])

print('----------------------------------------------')
print('Using the elbow method to find the optimal number of clusters')

from sklearn.cluster import KMeans
wcss = [] #Within-Cluster Sum of Squares
# WCSS) is calculated as the sum of the squared distances between each data point and 
# the centroid of the cluster to which it belongs. The goal of the K-Means algorithm is to 
# minimize WCSS, thereby creating clusters with tightly packed points.

#---
for i in range(1, 11): #from 1 to to 10 clusters (11 not included)
    #note: k-means++ tends to avoid the problem of random initialization trap, thus 
    # providing more accurate clustering results
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42) #42 is just because
    kmeans.fit(x)
    wcss.append(kmeans.inertia_) 
    #The inertia_ attribute is a float number representing the sum of squared distances of 
    # samples to their closest cluster center. It's a measure of how internally coherent 
    # clusters are. Lower values are better and zero is optimal
#---

print('wcss:')
print(wcss)

print('\nWhat is the best number of clusters?')
prev_wcss = wcss[0] * 1.30
for i, v in enumerate(wcss):
    # print(f'i:{i}, v:{v}, prev_wcss:{prev_wcss}, % prev_wcss:{v/prev_wcss}')

    if v/prev_wcss >= 0.9:
        print(f'    A potential good number of clusters is {i}')
        break
    prev_wcss = v

exit()





plt.plot(range(1, 11), wcss) #from 1 to 10 clusters and the WCSS associated with it
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

exit()

print('----------------------------------------------')
print('Training the K-Means model on the dataset')

kmeans = KMeans(n_clusters = 5 , init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(x)
print(y_kmeans)
exit()

print('----------------------------------------------')
print('Visualising the clusters')

plt.scatter(x[ y_kmeans == 0, 0], x[ y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')

plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(x[y_kmeans == 3, 0], x[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(x[y_kmeans == 4, 0], x[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()