import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
from sklearn.cluster import MeanShift
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

df = pd.read_excel('titanic.xls')
# make copy of original data frame so we can reference actual text rather than converted numbers
original_df = pd.DataFrame.copy(df)

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

df.drop(['ticket', 'home.dest'], 1, inplace=True)

# get rid of survived column as that would be cheating!
X = np.array(df.drop(['survived'], 1)).astype(float)
# scale data for optimisation
X = preprocessing.scale(X)
y = np.array(df['survived'])

clf = MeanShift()
clf.fit(X)


# analyze
labels = clf.labels_
cluster_centers = clf.cluster_centers_

# add new column to original data frame to fill in later
original_df['cluster_group'] = np.nan
# populate columns
for i in range(len(X)):
    # iloc can reference 'index' of the data frame (ie row)
    original_df['cluster_group'].iloc[i] = labels[i]

#
n_clusters_ = len(np.unique(labels))
survival_rates = {}
for i in range(n_clusters_):
    temp_df = original_df[ (original_df['cluster_group'] == float(i)) ]
    survival_cluster = temp_df[ (temp_df['survived'] == 1) ]
    # get survival rate
    survival_rate = len(survival_cluster) / len(temp_df)
    survival_rates[i] = survival_rate

print(survival_rates)

# analyze overview
print(original_df[ (original_df['cluster_group'] == 0)].describe())
