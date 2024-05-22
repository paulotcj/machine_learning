print('----------------------------------------------')
print('Hierarchical Clustering')

print('----------------------------------------------')
print('Importing the libraries')

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

print('----------------------------------------------')
print('Importing the dataset')

dataset = pd.read_csv('Mall_Customers.csv')
x = dataset.iloc[:, [3, 4]].values #import all rows, but only columns 3 and 4
# this is clustering, so we don't need to split the dataset into training and test sets
# also, because we plan to show this result in a 2D graph, we can only use 2 columns (2 dimensional plot)

print('x (sample 5 first rows):')
print(x[0:5])


print('----------------------------------------------')
print('Using the dendrogram to find the optimal number of clusters')

import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram( sch.linkage(x, method = 'ward') )

# import scipy.cluster.hierarchy as sch
# dendrogram = sch.dendrogram(sch.linkage(x, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()
exit()
print('----------------------------------------------')
print('Training the Hierarchical Clustering model on the dataset')
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(x)

print('----------------------------------------------')
print('Visualising the clusters')
plt.scatter(x[y_hc == 0, 0], x[y_hc == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(x[y_hc == 1, 0], x[y_hc == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(x[y_hc == 2, 0], x[y_hc == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(x[y_hc == 3, 0], x[y_hc == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(x[y_hc == 4, 0], x[y_hc == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()