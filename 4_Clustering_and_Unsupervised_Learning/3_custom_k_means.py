# build custom version of KMeans

# pick any two data points to start from
# measure every distance from every other points to those points
# whichever they are closer to, we classify them as belonging to that centroid's class
# take mean of both classes and that mean becomes new centroid
# repeat until centroid stops moving


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
#
# plt.scatter(X[:, 0], X[:, 1], s=150)
# plt.show()

colors = 10 * ["g", "r", "c", "b", "k"]



# custom KMeans
class K_Means:

    def __init__(self, k=2, tol=0.001, max_iter=300):
        self.k = k
        # tol: how much that centroid is going to move (in this case by percent change)
        self.tol = tol
        # how many iterations before ending
        self.max_iter = max_iter

    def fit(self, data):
        self.centroids = {}

        # set first two centroids as first two indexes of data array
        for i in range(self.k):
            self.centroids[i] = data[i]

        # optimisation process
        for i in range(self.max_iter):
            # reset classficiations each time the centroids change
            self.classifications = {}

            # keys will be centroids and values will be feature sets within those values
            for i in range(self.k):
                self.classifications[i] = []

            # populate list
            for featureset in data:
                # create list that is being populated with k number of values, with distances
                distances = [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classifications[classification].append(featureset)

            # create new dictionary with self.centroids to retain previous values
            prev_centroids = dict(self.centroids)

            for classification in self.classifications:
                # take array value and take average of all classifications we have (finding the mean of all the features of a class)
                # redefine centroid by reassigning self.centroids[classification]
                self.centroids[classification] = np.average(self.classifications[classification], axis=0)

            # innocent until proven guilty
            optimised = True

            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]

                # compare the two
                # if any centroids move more than our tolerance, we say we are not optimised
                if np.sum((current_centroid - original_centroid) / original_centroid * 100.0) > self.tol:
                    print(np.sum((current_centroid - original_centroid) / original_centroid * 100.0))
                    optimised = False


            if optimised:
                break


    def predict(self, data):
        distances = [np.linalg.norm(data - self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification



# create instance of our custom classifier
clf = K_Means()
# train
clf.fit(X)

for centroid in clf.centroids:
    plt.scatter(clf.centroids[centroid][0], clf.centroids[centroid][1], marker="o", color="k", s=150, linewidths=5)

for classification in clf.classifications:
    color = colors[classification]

    for featureset in clf.classifications[classification]:
        plt.scatter(featureset[0], featureset[1], marker="x", color=color, s=150, linewidths=5)


# test predictions
unknowns = np.array([[1, 3],
                     [8, 9],
                     [0, 3],
                     [5, 4],
                     [6, 4]
])

for unknown in unknowns:
    classficiation = clf.predict(unknown)
    plt.scatter(unknown[0], unknown[1], marker="*", color=colors[classification], s=150, linewidths=5)

plt.show()
