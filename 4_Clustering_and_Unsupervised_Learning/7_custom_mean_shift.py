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
              [9, 11],
              [8, 2],
              [10, 2],
              [9, 3]
])

# plt.scatter(X[:, 0], X[:, 1], s=150)
# plt.show()

colors = 10 * ["g", "r", "c", "b", "k"]



# custom Mean Shift class
class Mean_Shift:
    # NOTE: hard-coding the radius is prone to error! Better to use dynamic bandwidth!
    def __init__(self, radius=4):
        self.radius = radius

    def fit(self, data):
        centroids = {}

        # set initial centroids
        for i in range(len(data)):
            centroids[i] = data[i]

        while True:
            new_centroids = []

            for i in centroids:
                centroid = centroids[i]

                in_bandwidth = []

                # populate in_bandwidth with centroids within radius
                for featureset in data:
                    if np.linalg.norm(featureset - centroid) < self.radius:
                        in_bandwidth.append(featureset)

                # recalculate centroid using mean vector
                new_centroid = np.average(in_bandwidth, axis=0)
                new_centroids.append(tuple(new_centroid)) # tuple required for getting unique set of tuples (can't do that with list)

            # get unique elements from new_centroids list
            uniques = sorted(list(set(new_centroids)))

            # copy original centroids for reference later
            prev_centroids = dict(centroids)

            # define new centroids dictionary
            centroids = {}
            for i in range(len(uniques)):
                centroids[i] = np.array(uniques[i])

            # innocent until proven guilty
            optimised = True

            for i in centroids:
                # compare two arrays
                if not np.array_equal(centroids[i], prev_centroids[i]):
                    optimised = False

                if not optimised:
                    break

            if optimised:
                break

        # if optimised
        self.centroids = centroids

    def predict(self, data):
        pass



# create instance of Mean_Shift
clf = Mean_Shift()
clf.fit(X)

centroids = clf.centroids

# scatter data
plt.scatter(X[:, 0], X[:, 1], s=150)

# scatter centroids
for c in centroids:
    plt.scatter(centroids[c][0], centroids[c][1], color='k', marker='*', s=150)

plt.show()
