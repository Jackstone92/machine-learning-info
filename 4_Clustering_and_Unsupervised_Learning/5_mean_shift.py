# Mean Shift is a Hierarchical Clustering Algorithm

# unlike KMeans, the machine figures out how many clusters there should be and where they are
# every data point is considered a cluster center initially

# Radius bandwidth
# following process applied to every data point:
# each datapoint has a circular bandwidth around it. Might have feature sets within its radius
# take mean of all the datapoints within its radius
# new cluster center is that mean and with it comes a new bandwidth
# again, take all feature sets within that bandwidth and take mean of all of those
# when no other new features found within bandwidth and cluster center doesnt move any more, it is optimised
# therefore, all datapoints converge until they are fully optimised

# can have different levels and can assign different weights for each


import numpy as np
from sklearn.cluster import MeanShift
from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import style
style.use('ggplot')


# create starting centers so we can create sample data
centers = [[1,1,1],
           [5,5,5],
           [3,10,10]
]

# create sample data
X, _ = make_blobs(n_samples=100, centers=centers, cluster_std=1)

# use mean shift
ms = MeanShift()
ms.fit(X)

labels = ms.labels_
cluster_centers = ms.cluster_centers_
print(cluster_centers)
n_clusters_ = len(np.unique(labels))
print("Number of estimated clusters:", n_clusters_)

colors = 10*['r', 'g', 'b', 'c', 'k', 'y', 'm']
fig = plt.figure()
# create 3D graph
ax = fig.add_subplot(111, projection='3d')

# plot points
for i in range(len(X)):
    ax.scatter(X[i][0], X[i][1], X[i][2], c=colors[labels[i]], marker='o')

ax.scatter(cluster_centers[:,0], cluster_centers[:,1], cluster_centers[:,2], marker='x', color='k', s=150, linewidths=5, zorder=10)

plt.show()
