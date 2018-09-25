# Everything hinges on Euclidean Distance (Father of Geometry)
# Square root of (Sum to n where i starts off being equal to 1 (qi - pi)^2)

# eg. breaking down into simple mathematics:
# q = (1, 3)
# p = (2, 5)
# square root of (q1 - p1)^2 + (q2 - p2)^2
# square root of (1 - 2)^2 + (3 - 5)^2

# converting to custom python code:
# from math import sqrt
#
# plot1 = [1, 3]
# plot2 = [2, 5]
#
# euclidean_distance = sqrt( (plot1[0] - plot2[0])**2 + (plot1[1] - plot2[1])**2 )
#
# print(euclidean_distance)


# using numpy
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import warnings
from matplotlib import style
from collections import Counter
style.use('fivethirtyeight')

# dataset of class k, which is a list of features
dataset = {
    'k': [ [1,2], [2,3], [3,1] ],
    'r': [ [6,5], [7,7], [8,6] ]
}
new_features = [5,7]

def k_nearest_neighbours(data, predict, k=3):
    # can't have more classes than k
    if len(data) >= k:
        warnings.warn('K is set to a value less than total voting groups! Idiot!')

    # Have to compare euclidean distance between every feature in dataset...
    # This is the problem with K nearest neighbours!

    distances = []
    for class_ in data:
        for features in data[class_]:
            # euclidean distance written out using numpy:
            # euclidean_distance = np.sqrt(np.sum((np.array(features) - np.array(predict))**2))

            # use numpy linear algebra to calculate euclidean distance
            euclidean_distance = np.linalg.norm(np.array(features) - np.array(predict))
            distances.append([euclidean_distance, class_])

    votes = [i[1] for i in sorted(distances)[:k]]
    print(Counter(votes).most_common(1))
    vote_result = Counter(votes).most_common(1)[0][0]
    # confidence = how many / k
    confidence = Counter(votes).most_common(1)[0][1] / k


    return vote_result


result = k_nearest_neighbours(dataset, new_features, k=3)
print(result)


# visualise data:
# for i in dataset:
#     for feature in dataset[i]:
#         plt.scatter(feature[0], feature[1], s=100, color=feature)
# converted to 1 liner:
[[plt.scatter(feature[0], feature[1], s=100, color=i) for feature in dataset[i]] for i in dataset]
plt.scatter(new_features[0], new_features[1], s=200, color=result)
plt.show()
