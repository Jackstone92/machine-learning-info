# Two main forms of clustering:
# Flat clustering and Hierarchical clustering
# The machine is given feature sets and the machine searches for clusters

# Flat clustering:
# You tell the machine to find n clusters

# Hierarchical clustering:
# The machine figures out the groups and how many groups there ought to be


# K-Means algorithm (based on flat clustering)
# where k is the number of clusters you want

# Mean-Shift (based on hierarchical clustering)


# K-Means
# Choose centroids (centres of clusters) eg. first k items of feature set or shuffled items
# classify any new feature sets based on euclidian proximity to centroids
# create new centroids based on the mean of all nodes' proximity to original centroids
# repeat until centroid stops moving -> then you have your clusters!

# Main issue with K-Means: Mouse Data issue and scaling!

# Python example:
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
from sklearn.cluster import KMeans

X = np.array([[1, 2],
              [1.5, 1.8],
              [5, 8],
              [8, 8],
              [1, 0.6],
              [9, 11]
])

# plt.scatter(X[:, 0], X[:, 1], s=150)
# plt.show()

# define classifier
clf = KMeans(n_clusters=2)

clf.fit(X)

# access centroids
centroids = clf.cluster_centers_
# array of labels of our features
labels = clf.labels_

colors = 10 * ["g.", "r.", "c.", "b.", "k."]

for i in range(len(X)):
    # i is our index value
    # plot
    plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize=10)

plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=150, linewidths=5)
plt.show()
