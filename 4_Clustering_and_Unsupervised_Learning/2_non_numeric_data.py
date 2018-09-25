import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing
import pandas as pd

# how titanic dataset is laid out:
'''
pclass Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd)
survival Survival (0 = No; 1 = Yes)
name Name
sex Sex
age Age
sibsp Number of Siblings/Spouses Aboard
parch Number of Parents/Children Aboard
ticket Ticket Number
fare Passenger Fare (British Pound)
cabin Cabin
embarked Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)
boat Lifeboat
body Body Identification Number
home.dest Home/Destination
'''


# can use this to determine if a person would survive or die on the titanic

# use KMeans to separate into two groups:
# did they survive or not? How accurate are these groups?

# Machine learning requires you to have numerical data
# Convert non-numerical data into numerical data
# eg. sex column (female, male) and convert to 0 for female and 1 for male
# thereby we assign a unique id for each!


# read into data frame using pandas
df = pd.read_excel('titanic.xls')

# drop unnecessary columns
df.drop(['body', 'name'], 1, inplace=True)
df.convert_objects(convert_numeric=True) # depricated... try to find another way!
df.fillna(0, inplace=True)


# handle non-numerical data
def handle_non_numerical_data(df):
    columns = df.columns.values

    # go through each column
    for column in columns:
        text_digit_vals = {}

        def convert_to_int(val):
            return text_digit_vals[val]

        # if column is not a number
        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            # convert to list
            column_contents = df[column].values.tolist()
            # create unique set from list
            unique_elements = set(column_contents)
            x = 0
            # populate dictionary
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x += 1

            # reset the values of df[column] by mapping
            df[column] = list(map(convert_to_int, df[column]))

    return df


df = handle_non_numerical_data(df)
# print(df.head())

df.drop(['boat', 'sex'], 1, inplace=True)

# get rid of survived column as that would be cheating!
X = np.array(df.drop(['survived'], 1)).astype(float)
# scale data for optimisation
X = preprocessing.scale(X)
y = np.array(df['survived'])

clf = KMeans(n_clusters=2)
clf.fit(X)

# magic
correct = 0
for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = clf.predict(predict_me)

    if prediction[0] == y[i]:
        correct += 1

print(correct / len(X))
