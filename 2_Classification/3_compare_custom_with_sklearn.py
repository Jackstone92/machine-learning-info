import numpy as np
from math import sqrt
import warnings
from collections import Counter
import pandas as pd
import random


def k_nearest_neighbours(data, predict, k=3):
    # can't have more classes than k
    if len(data) >= k:
        warnings.warn('K is set to a value less than total voting groups! Idiot!')

    distances = []
    for class_ in data:
        for features in data[class_]:
            euclidean_distance = np.linalg.norm(np.array(features) - np.array(predict))
            distances.append([euclidean_distance, class_])

    votes = [i[1] for i in sorted(distances)[:k]]
    # print(Counter(votes).most_common(1))
    vote_result = Counter(votes).most_common(1)[0][0]
    # confidence = how many / k
    confidence = Counter(votes).most_common(1)[0][1] / k

    return vote_result, confidence



accuracies = []

for i in range(25):
    df = pd.read_csv('breast-cancer-wisconsin.data.txt')
    df.replace('?', -99999, inplace=True)
    df.drop(['id'], 1, inplace=True)

    # convert everything in this data frame to a float
    # some of the values were treated as strings...
    full_data = df.astype(float).values.tolist()

    # shuffle the data
    random.shuffle(full_data)

    # setup test and train data
    test_size = 0.2
    train_set = {2: [], 4: []}
    test_set = {2: [], 4: []}
    train_data = full_data[:-int(test_size * len(full_data))] # first 80% of the data
    test_data = full_data[-int(test_size * len(full_data)):] # last 20% of the data

    # populate train_set and test_set dictionaries
    for i in train_data:
        train_set[i[-1]].append(i[:-1])

    for i in test_data:
        test_set[i[-1]].append(i[:-1])

    # run custom k_nearest_neighbours method
    correct = 0
    total = 0

    for class_ in test_set:
        for data in test_set[class_]:
            vote, confidence = k_nearest_neighbours(train_set, data, k=5)

            # check if right
            if class_ == vote:
                correct += 1
            # else:
            #     print(confidence)
            total += 1

    # print('Accuracy: ', correct/total)
    accuracies.append(correct/total)

print(sum(accuracies) / len(accuracies))


# what are the differences?
# sklearn is waaay faster!
# KNeighborsClassifier can be threaded using n_jobs=-1
# KNeighborsClassifier uses idea of radius rather than calculating every single euclidian distance for every feature
# However, accuracies are basically the same. Just speed improvement using sklearn
