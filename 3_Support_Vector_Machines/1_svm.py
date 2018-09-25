# What is a support vector machine?

# High-level intuition of the machine:
# We are in vector space
# uses binary classifier to separate into two groups (positive and negative)
# Aim is to find decision boundary or best separating hyperplane
# Once aquired, we can take in unknown data and classify it according to the decision boundary
# We want to use it against linear data


# example using sklearn:
import numpy as np
# import svm
from sklearn import preprocessing, cross_validation, svm
import pandas as pd



df = pd.read_csv('breast-cancer-wisconsin.data.txt')
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)

# define Xs (features) and ys (labels or class)
X = np.array(df.drop(['class'], 1))
y = np.array(df['class'])

# cross-validation
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

# define classifier - support vector classifier
clf = svm.SVC()
clf.fit(X_train, y_train)

# test data
accuracy = clf.score(X_test, y_test)
print("Accuracy: ", accuracy)
