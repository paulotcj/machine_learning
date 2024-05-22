print('----------------------------------------------')
print('Hierarchical Clustering')

# Hierarchical Clustering
# Advantages:
# Dendrograms: Hierarchical clustering produces a dendrogram, which is a tree-like diagram 
#  that shows the arrangement of the clusters. This is useful for visualizing the structure 
#  and determining the number of clusters.
#
# No Need to Predefine k: Unlike K-Means++, hierarchical clustering does not require the number 
#  of clusters to be specified in advance. You can cut the dendrogram at the desired level to 
#  obtain the required number of clusters.
#
# Flexibility: Can capture complex cluster structures and works well with non-spherical clusters.
#
# Robustness to Outliers: Less sensitive to outliers compared to K-Means++.
#
# Disadvantages:
# Computational Complexity: Hierarchical clustering can be computationally intensive, especially for 
#  large datasets, because it has a time complexity of O(n^3) and a space complexity of O(n^2).
#
# Scalability: Due to its computational demands, hierarchical clustering is less scalable than 
#  K-Means++, making it less suitable for very large datasets.
#
# Merging and Splitting Issues: Once a merge or split is made in hierarchical clustering, it cannot 
#  be undone, which can lead to suboptimal clusters if early decisions are incorrect.
#
#--------------
#
# When to Use K-Means++:
# Large datasets where computational efficiency is important.
# When you have a good estimate of the number of clusters.
# When clusters are expected to be spherical and of similar size.
# Applications like market segmentation, image compression, and document clustering.
#
#
# When to Use Hierarchical Clustering:
# Smaller datasets where computational complexity is manageable.
# When the underlying structure of the data is unknown and you want to explore it.
# When you need to visualize the clustering process and determine the number of clusters from the dendrogram.
# Applications like gene expression analysis, customer segmentation with hierarchical relations, 
#  and any scenario where visualizing the cluster hierarchy is beneficial.

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

# 'ward' is used, which minimizes the variance of the clusters being merged.
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram( sch.linkage(x, method = 'ward') )

plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()

print('----------------------------------------------')
print('Training the Hierarchical Clustering model on the dataset')

from sklearn.cluster import AgglomerativeClustering
hierarchical_clustering = AgglomerativeClustering(n_clusters = 5 , metric = 'euclidean', linkage = 'ward')
y_hc = hierarchical_clustering.fit_predict(x)



print('----------------------------------------------')
print('Visualising the clusters')

# All elements below will be shown in their correct postion, but we slice the data according 
# to the cluster - so we can show the clusters in different colors

# y_kmeans == 0 -> this means where y_kmeans is 0, it will return True, otherwise False
# so for x[ y_kmeans == 0, 0] - this means we generate a mask with the same size of X
# where each element is True or False. If it is True, we select the element in the same 
# position in X
# And with x[ y_kmeans == 0, 0] we select the first column

# x[y_kmeans == 0, 0] means that we are selecting all the rows in x that belong to cluster 0
# x[y_kmeans == 0, 1] means that we are selecting all the rows in x that belong to cluster 0
# s = 100 is the size of the point, c is the color, label is the legend

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