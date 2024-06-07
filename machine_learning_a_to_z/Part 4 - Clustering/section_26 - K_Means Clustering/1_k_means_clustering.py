print('----------------------------------------------')
print('K-Means Clustering')

# K-Means++ Clustering
# Advantages:
# Efficiency: K-Means++ is computationally efficient, especially for large datasets. 
#  The algorithm scales well with the number of data points.
#
# Convergence: K-Means++ improves the initialization of centroids, leading to faster and 
#  often more accurate convergence compared to traditional K-Means.
#
# Interpretability: The clusters produced by K-Means++ are often easy to interpret, especially 
#  when the number of clusters is small.
#
# Applicability: Works well when clusters are spherical and of similar size.
#
# Disadvantages:
# Predefined Number of Clusters: K-Means++ requires the number of clusters (k) to be specified in 
#  advance, which can be a limitation if the optimal number of clusters is not known.
#
# Sensitivity to Outliers: K-Means++ is sensitive to outliers, which can significantly affect the 
#  centroids and the resulting clusters.
#
# Assumption of Convex Shapes: It assumes that clusters are convex, making it less effective for 
#  datasets with irregular or elongated cluster shapes.
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

#the previous WCSS when we start is undefined, so we start with a big number
prev_wcss = float('inf')
for i, v in enumerate(wcss):
    # print(f'i:{i}, v:{v}, prev_wcss:{prev_wcss}, % prev_wcss:{v/prev_wcss}')

    if v/prev_wcss >= 0.9: # if the WCSS is 90% of the previous WCSS, then we are not improving much
        print(f'    A potential good number of clusters is {i}')
        break
    prev_wcss = v


exit()

plt.plot(range(1, 11), wcss) #from 1 to 10 clusters and the WCSS associated with it
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()



print('----------------------------------------------')
print('Training the K-Means model on the dataset')

kmeans = KMeans(n_clusters = 5 , init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(x)
print('Note: Each element in y_kmeans is the cluster number that the element in x belongs to')
print('  So if we see that y_kmeans[0] = 2, it means that the first element in x belongs to cluster 2')
print(y_kmeans)


print('----------------------------------------------')
print('Visualising the clusters')

print('y_kmeans == 0')
print(y_kmeans == 0)
print('----')
print('x[ y_kmeans == 0, 0]')
print(x[ y_kmeans == 0, 0])
print('----')
print('centroids:')
print(kmeans.cluster_centers_)

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
plt.scatter(x[ y_kmeans == 0, 0], x[ y_kmeans == 0, 1], s = 70, c = 'red', label = 'Cluster 1')
plt.scatter(x[ y_kmeans == 1, 0], x[ y_kmeans == 1, 1], s = 70, c = 'blue', label = 'Cluster 2')
plt.scatter(x[ y_kmeans == 2, 0], x[ y_kmeans == 2, 1], s = 70, c = 'green', label = 'Cluster 3')
plt.scatter(x[ y_kmeans == 3, 0], x[ y_kmeans == 3, 1], s = 70, c = 'cyan', label = 'Cluster 4')
plt.scatter(x[ y_kmeans == 4, 0], x[ y_kmeans == 4, 1], s = 70, c = 'magenta', label = 'Cluster 5')

#get the coordinates of all centroids

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=40, c='yellow', edgecolors='black', label='Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()