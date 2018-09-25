# how close is a new point from k other nearest points?
# k should be an odd number so as to avoid equal splits

# The larger the dataset the worse this algorithm runs because it calculates Euclidian distance between each of the nodes

# website for datasets: http://archive.ics.uci.edu/ml/datasets.html

import numpy as np
from sklearn import preprocessing, cross_validation, neighbors
import pandas as pd


accuracies = []

for i in range(25):
    # load in txt file (treated like a csv) into dataframe
    df = pd.read_csv('breast-cancer-wisconsin.data.txt')
    # missing data is denoted by a "?" - we need to replace that data to be an outlier
    df.replace('?', -99999, inplace=True)

    # we don't need the id column as it has nothing to do in determining if case has breast cancer
    df.drop(['id'], 1, inplace=True)

    # define Xs (features) and ys (labels or class)
    X = np.array(df.drop(['class'], 1))
    y = np.array(df['class'])


    # cross-validation
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)


    # define classifier
    clf = neighbors.KNeighborsClassifier()
    clf.fit(X_train, y_train)


    # test data
    accuracy = clf.score(X_test, y_test)
    # print("Accuracy: ", accuracy)


    # make prediction
    # example_measures = np.array([ [4,2,1,1,1,2,3,2,1] ])
    # # must reshape to ensure dimensions of numpy array. Easiest way is to use len()
    # example_measures = example_measures.reshape(len(example_measures), -1)
    #
    # prediction = clf.predict(example_measures)
    # print("Prediction: ", prediction)

    accuracies.append(accuracy)

print(sum(accuracies) / len(accuracies))
