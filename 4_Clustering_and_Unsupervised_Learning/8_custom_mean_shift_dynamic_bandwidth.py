import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_blobs
import random


centers = random.randrange(2, 5)


X, y = make_blobs(n_samples=100, centers=centers, n_features=2)


# X = np.array([[1, 2],
#               [1.5, 1.8],
#               [5, 8],
#               [8, 8],
#               [1, 0.6],
#               [9, 11],
#               [8, 2],
#               [10, 2],
#               [9, 3]
# ])

colors = 10 * ["g", "r", "c", "b", "k"]



# custom Mean Shift class with Dynamic Bandwidth
class Mean_Shift:

    def __init__(self, radius=None, radius_norm_step=100):
        self.radius = radius
        self.radius_norm_step = radius_norm_step

    def fit(self, data):

        if self.radius == None:
            # find centroid of all data
            all_data_centroid = np.average(data, axis=0)
            # find magnitude
            all_data_norm = np.linalg.norm(all_data_centroid)
            # use to determine decent overall radius
            self.radius = all_data_norm / self.radius_norm_step

        centroids = {}

        # set initial centroids
        for i in range(len(data)):
            centroids[i] = data[i]

        # define weights and reverse list
        weights = [i for i in range(self.radius_norm_step)][::-1]

        while True:
            new_centroids = []

            for i in centroids:
                centroid = centroids[i]
                in_bandwidth = []

                for featureset in data:
                    # calc full distance
                    distance = np.linalg.norm(featureset - centroid)

                    if distance == 0:
                        distance = 0.000000001

                    weight_index = int(distance / self.radius)

                    if weight_index > self.radius_norm_step - 1:
                        weight_index = self.radius_norm_step - 1

                    to_add = (weights[weight_index] **2 ) * [featureset]
                    # add list
                    in_bandwidth += to_add


                # recalculate centroid using mean vector
                new_centroid = np.average(in_bandwidth, axis=0)
                new_centroids.append(tuple(new_centroid)) # tuple required for getting unique set of tuples (can't do that with list)

            # get unique elements from new_centroids list
            uniques = sorted(list(set(new_centroids)))

            to_pop = []

            for i in uniques:
                # we're not inspecting centroids in radius of i since i will be popped
                if i in to_pop:
                    pass

                for ii in uniques:
                    if i == ii:
                        pass
                    # skipping already-added centroids
                    elif np.linalg.norm(np.array(i)-np.array(ii)) <= self.radius and ii not in to_pop:
                        to_pop.append(ii)
                        break

            for i in to_pop:
                try:
                    uniques.remove(i)
                except:
                    pass


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

        # add classifications
        self.classifications = {}

        for i in range(len(self.centroids)):
            self.classifications[i] = []

        for featureset in data:
            distances = [np.linalg.norm(featureset - self.centroids[centroid]) for centroid in self.centroids]
            classification = distances.index(min(distances))
            self.classifications[classification].append(featureset)


    def predict(self, data):
            distances = [np.linalg.norm(featureset - self.centroids[centroid]) for centroid in self.centroids]
            classification = distances.index(min(distances))
            self.classifications[classification].append(featureset)


# create instance of Mean_Shift
clf = Mean_Shift()
clf.fit(X)

centroids = clf.centroids

# scatter featuresets
for classification in clf.classifications:
    color = colors[classification]
    for featureset in clf.classifications[classification]:
        plt.scatter(featureset[0], featureset[1], marker='x', color=color, s=150, linewidths=5)

# scatter centroids
for c in centroids:
    plt.scatter(centroids[c][0], centroids[c][1], color='k', marker='*', s=150)

plt.show()
