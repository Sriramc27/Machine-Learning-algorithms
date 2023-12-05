import os
import pandas as pd
import numpy as np
import seaborn as sns
import collections
from sklearn import preprocessing
# import class for hyperparameter tuning
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import tensorflow as tf
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder
# import class for z-score normalization
import scipy.stats as stats
# import class for min-max scaler
from sklearn.preprocessing import MinMaxScaler

largecsv = os.getcwd() + "\\Combined_MNIST_70k.csv"
smallcsv = os.getcwd() + "\\Dataset_10k_1.csv"


# use this function to split the dataset in a by the labels in b
def get_clusters(a, b):
    s = np.argsort(b)
    return np.split(a[s], np.unique(b[s], return_index=True)[1][1:])


# violin plots for any dataset
def draw_violin(x, y, t, n):
    clt = get_clusters(x, y)
    plt.figure()
    plt.violinplot(clt, showmedians=True)
    # Create an axes instance
    # ax = fig.add_axes([0, 0, 1, 1])

    # fig, ax = plt.subplots()

    # Set plot labels and title
    plt.xlabel('MNIST class')
    plt.ylabel('Value')
    plt.title('Violin Plot for ' + n)

    # np.random.seed(2)
    # Set x-axis tick labels
    plt.xticks(np.arange(1, len(t) + 1), t)

    # Create the boxplot
    # bp = ax.violinplot(x)

    # Show the plot
    plt.show()


def xgb_score(x_trn, y_trn, x_tst, y_tst, r):
    xg_clf = XGBClassifier(learning_rate=0.3, n_estimators=150, max_depth=6, min_child_weight=1, gamma=0,
                           reg_lambda=1, subsample=1, colsample_bytree=1, scale_pos_weight=1,
                           objective='multi:softprob', num_class=10, random_state=r)
    xg_clf.fit(x_trn, y_trn)
    xg_predict = xg_clf.predict(x_tst)
    # calculate accuracy for test data
    xg_scores_total = xg_clf.score(x_tst, y_tst)
    xg_scores = cross_val_score(xg_clf, x_trn, y_trn, cv=5)
    for s in range(xg_scores.shape[0]):
        xg_scores_total += xg_scores[s]

    return xg_scores_total / (xg_scores.shape[0] + 1)


if __name__ == "__main__":
    largepd = pd.read_csv(largecsv, header=0)  # load large csv to memory
    smallpd = pd.read_csv(smallcsv, header=0)  # load small csv to memory
    targets = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # define 10 classes for single digits

    X_large = largepd.drop(largepd.columns[[0, 1]], axis=1)  # drop first 2 columns
    y_large = largepd['0']  # these are the labels
    X_small = smallpd.drop(smallpd.columns[[0, 1]], axis=1)  # drop first 2 columns
    y_small = smallpd['0']  # these are the labels
    X_large_mean = X_large.mean(axis=1)
    X_small_mean = X_small.mean(axis=1)

    # use z-score normalization
    X_z_sm = stats.zscore(X_small)
    X_z_sm_mn = X_z_sm.mean(axis=1)
    X_z_lg = stats.zscore(X_large)
    X_z_lg_mn = X_z_lg.mean(axis=1)

    # user min-max normalization
    scaler = MinMaxScaler()
    X_minmax_sm = scaler.fit_transform(X_small)
    X_minmax_lg = scaler.fit_transform(X_large)
    X_minmax_sm_mn = X_minmax_sm.mean(axis=1)
    X_minmax_lg_mn = X_minmax_lg.mean(axis=1)

    # get the counts
    unique, counts = np.unique(y_large.shape[0], return_counts=True)
    counterlarge = collections.Counter(y_large)
    print(counterlarge)
    countersmall = collections.Counter(y_small)
    print(countersmall)

    # draw violin plots with not-normalized data
    draw_violin(X_small_mean, y_small, targets, 'Small dataset not-normalized')
    draw_violin(X_large_mean, y_large, targets, 'Large dataset not-normalized')

    # violin plots with z-score normalized data
    draw_violin(X_z_sm_mn, y_small, targets, 'Small dataset z-score')
    draw_violin(X_z_lg_mn, y_large, targets, 'Large dataset z-score')

    # violin plots with min-max normalized data
    draw_violin(X_minmax_sm_mn, y_small, targets, 'Small dataset min-max')
    draw_violin(X_minmax_lg_mn, y_large, targets, 'Large dataset min-max')



