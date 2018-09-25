# idea behind regression is:
# take continuous data and try and find the best fit line
# try and model data
# y = mx + c -> try and find out what m and c are

# this example will look at trying to model stock pricing

import pandas as pd
# use quandl to get stock info
import quandl, math, datetime
# numpy to allow us to use arrays (python doesn't actually use arrays)
import numpy as np

# preprocessing: use scaling (on features to get them between -1 and 1)
# cross_validation: create training and testing samples (shuffle data for you to help with statistics and helps to separate data)
# svm (support vector machine): can use svm to do regression
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression

# use matplotlib for graphing
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

# use for pickling
import pickle


# create data frame for google stocks
df = quandl.get('WIKI/GOOGL')

df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]

# define special relationships and use them as features:
# high-low => % volitility
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0

# daily move or percent change => new - old / old * 100
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0


# define new data frame with only features we care about
# what actually affects price?
#           Price         x           x              x
df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]


# create variable to use for forecasting
forecast_col = 'Adj. Close'
# use pandas to fill and N/A values
# in machine learning, you can't work with NAN data -> do this to be treated as an outlier
df.fillna(-99999, inplace=True)

# create forecast out
# try to predict 10% (ie. 0.1) of the dataset
forecast_out = int(math.ceil(0.1*len(df)))
print("Forecast_out: ", forecast_out)

# print(df.head())

# define label
# shift columns negatively so each row's adj. close might predict 1% in the future
df['label'] = df[forecast_col].shift(-forecast_out)


# define x and y
# features will be X and labels will be y
X = np.array(df.drop(['label'], 1)) # return new data frame with everything except label
# now going to scale X: constantly scaling alongside all our other values (NOTE: extra processing time required!)
X = preprocessing.scale(X)
# get values from the point of -forecast_out -> this is what we will predict against
X_lately = X[-forecast_out:] # this is the last 30 days (10%)
# trim X to the point of -forecast_out
X = X[:-forecast_out] # (90%)

df.dropna(inplace=True)
y = np.array(df['label']) # return new data frame with just the label

# print(len(X), len(y))


# create training and testing set
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

# define classifier
# find n_jobs parameter to run multi-threaded
clf = LinearRegression(n_jobs=-1)
# clf = svm.SVR() # if using svm algorithm -> easy to switch between algorithms!
# use classifier to fit features and labels
clf.fit(X_train, y_train)

# save the classifier to avoid doing the training step
with open('linearregression.pickle', 'wb') as file:
    pickle.dump(clf, file)

# use the classifier from pickle
pickle_in = open('linearregression.pickle', 'rb')
# redefine classifier as pickle file
clf = pickle.load(pickle_in)

# test classifier and provide accuracy
# accuracy is the squared error
accuracy = clf.score(X_test, y_test)
# print("Accuracy: ", accuracy)


# now to predict stuff based on the X data!
forecast_set = clf.predict(X_lately)

print(forecast_set, accuracy, forecast_out)
# reset forcast value!!! We will repopulate.
df['Forecast'] = np.nan

# find out what the last date was: forecast_set doesn't know what the date is!!
last_date = df.iloc[-1].name
last_unix_value = last_date.timestamp()
one_day = 86400 # number of seconds in a day
next_unix_value = last_unix_value + one_day

# populate dataframe with new dates and forecast values
# iterate through forecast set taking each forecast and day and then setting those as the values in the data frame
for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix_value)
    next_unix_value += one_day
    # df.loc: the date and is used as the index
    # np.nan: set everything to nan
    # + [i]: add the forecast to the very end
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]

print(df.tail())

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()


# pickling is the serialisation of a python object eg. dictionary or classifier
# use pickling to save state of the classifier so we don't have to train everytime we want to make a prediction
# wouldn't want to
