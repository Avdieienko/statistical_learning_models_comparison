import pandas as pd
import numpy as np

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt


def correlation(estimator, X, y):
    estimator.fit(X,y)
    prediction = estimator.predict(X)
    return r2_score(y, prediction)


def accuracy(estimator, X, y):
    estimator.fit(X, y)
    prediction = estimator.predict(X)
    return accuracy_score(y, prediction)


def importance(covariates,target, estimator):
    estimator.fit(df[covariates], df[target])
    return sorted(list(zip(covariates, estimator.feature_importances_)), key=lambda tup: -tup[1])


df = pd.read_csv("https://courses.edx.org/asset-v1:HarvardX+PH526x+2T2019+type@asset+block@movie_data.csv", index_col=0)
profitable = np.zeros(df.shape[0])
for i in range(df.shape[0]):
    if df["revenue"].iloc[i]>df["budget"].iloc[i]:
        profitable[i] = 1

df["profitable"] = profitable

# regression_target = "revenue"
# classification_target = "profitable"


df = df.replace(np.inf, np.nan)
df = df.replace(-np.inf, np.nan)
df = df.dropna()


genres = list()
for i in df.genres:
    data = i.split(", ")
    for j in data:
        if j not in genres:
            genres.append(j)

for genre in genres:
    df[genre] = df["genres"].str.contains(genre).astype(int)

# continuous_covariates = ['budget', 'popularity', 'runtime', 'vote_count', 'vote_average']
# outcomes_and_continuous_covariates = continuous_covariates + [regression_target, classification_target]
# plotting_variables = ['budget', 'popularity', regression_target]
#
# axes = pd.plotting.scatter_matrix(df[plotting_variables], alpha=0.15,
#        color=(0,0,0), hist_kwds={"color":(0,0,0)}, facecolor=(1,0,0))
# print(df[outcomes_and_continuous_covariates].skew())
# plt.show()

# Transform right skewed variables
right_skewed = ['budget', 'popularity','runtime','vote_count','revenue']
for skew in right_skewed:
    df[skew] = df[skew].apply(lambda x: np.log10(1+x))

# Choose films with only positive revenue
df = df[df.revenue>0]

# Saving covariates and outcomes
regression_target = 'revenue'
classification_target = 'profitable'
all_covariates = ['budget', 'popularity', 'runtime', 'vote_count', 'vote_average', 'Action', 'Adventure', 'Fantasy',
                  'Science Fiction', 'Crime', 'Drama', 'Thriller', 'Animation', 'Family', 'Western', 'Comedy', 'Romance',
                  'Horror', 'Mystery', 'War', 'History', 'Music', 'Documentary', 'TV Movie', 'Foreign']
regression_outcome = df[regression_target]
classification_outcome = df[classification_target]
covariates = df[all_covariates]

# Creating classification models
linear_regression = LinearRegression()
logistic_regression = LogisticRegression()
forest_regression = RandomForestRegressor(max_depth=4, random_state=0)
forest_classifier = RandomForestClassifier(max_depth=4, random_state=0)

# Calculating score for each model
linear_regression_scores = cross_val_score(linear_regression,covariates,regression_outcome, cv=10, scoring= correlation)
forest_regression_scores = cross_val_score(forest_regression,covariates,regression_outcome, cv=10, scoring= correlation)
logistic_regression_scores = cross_val_score(logistic_regression,covariates,classification_outcome, cv=10, scoring= accuracy)
forest_classification_scores = cross_val_score(forest_classifier,covariates,classification_outcome, cv=10, scoring= accuracy)


# Print the importance of each covariate in the random forest regression.
print(importance(all_covariates,classification_target, forest_classifier))


plt.figure(figsize=(8,4))
plt.subplot(121)
plt.scatter(linear_regression_scores, forest_regression_scores)
plt.plot((0, 1), (0, 1), 'k-')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel("Linear Regression Score")
plt.ylabel("Forest Regression Score")


plt.subplot(122)
plt.scatter(logistic_regression_scores, forest_classification_scores)
plt.plot((0, 1), (0, 1), 'k-')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel("Logistic Regression Score")
plt.ylabel("Forest Classification Score")

plt.suptitle("Classification accuracy comparison")

plt.show()



