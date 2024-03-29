import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
import pandas as pd
from sklearn import preprocessing, cross_validation
# from sklearn.cluster import KMeans



colors = 10 * ["g", "r", "c", "b", "k"]



# custom KMeans
class K_Means:

    def __init__(self, k=2, tol=0.001, max_iter=300):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, data):
        self.centroids = {}
        for i in range(self.k):
            self.centroids[i] = data[i]

        for i in range(self.max_iter):
            self.classifications = {}
            for i in range(self.k):
                self.classifications[i] = []

            for featureset in data:
                # create list that is being populated with k number of values, with distances
                distances = [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classifications[classification].append(featureset)

            prev_centroids = dict(self.centroids)

            for classification in self.classifications:
                self.centroids[classification] = np.average(self.classifications[classification], axis=0)

            optimised = True

            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]

                if np.sum((current_centroid - original_centroid) / original_centroid * 100.0) > self.tol:
                    print(np.sum((current_centroid - original_centroid) / original_centroid * 100.0))
                    optimised = False


            if optimised:
                break

    def predict(self, data):
        distances = [np.linalg.norm(data - self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification



# using titanic data
df = pd.read_excel('titanic.xls')
df.drop(['body', 'name'], 1, inplace=True)
df.convert_objects(convert_numeric=True)
# print(df.head())
df.fillna(0, inplace=True)

def handle_non_numerical_data(df):
    columns = df.columns.values

    for column in columns:
        text_digit_vals = {}
        def convert_to_int(val):
            return text_digit_vals[val]

        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x += 1

            df[column] = list(map(convert_to_int, df[column]))

    return df


df = handle_non_numerical_data(df)
# print(df.head())

df.drop(['ticket', 'home.dest'], 1, inplace=True)

# get rid of survived column as that would be cheating!
X = np.array(df.drop(['survived'], 1)).astype(float)
# scale data for optimisation
X = preprocessing.scale(X)
y = np.array(df['survived'])


# clf = KMeans(n_clusters=2)
clf = K_Means()
clf.fit(X)

# magic
correct = 0
for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = clf.predict(predict_me)

    if prediction == y[i]:
        correct += 1

print(correct / len(X))
