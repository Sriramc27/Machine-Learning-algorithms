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



def xgb_score(x_trn, x_tst, y_trn, y_tst, norm_type, r):
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

def Xg_bar_plots(results_dict, normalization_types, random_states):
    ## Plotting for both dataset
    # Create a bar plot
    bar_width = 0.25
    index = np.arange(len(random_states))

    for norm_type in normalization_types:
        accuracies = [results_dict[r][norm_type] for r in random_states]
        plt.bar(index, accuracies, width=bar_width, label=norm_type.capitalize())
        index = index + bar_width

    plt.xlabel('Random State')
    plt.ylabel('Accuracy')
    plt.title('Classification Results (Test Accuracy: Average of 5 test accuracies)')
    plt.xticks(index - bar_width * (len(normalization_types) / 2), random_states)
    plt.legend(title='Normalization Type')
    plt.show()



if __name__ == "__main__":
    largecsv = os.getcwd() + "\\Combined_MNIST_70k.csv"
    smallcsv = os.getcwd() + "\\Dataset_10k_1.csv"
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

    random_states = [0, 10, 20, 30, 40]
    normalization_types = ['unnormalized', 'z-score', 'min-max']
    results_dict_small = {}
    results_dict_large = {}

    ## Small Dataset
    for r in random_states:
        # # # Split the data
        results_dict_small[r] = {}
        for norm_type in normalization_types:
            if norm_type == 'unnormalized':
                x_train, x_test, y_train, y_test = train_test_split(X_small, y_small, test_size=0.2, random_state=42, stratify=y_small)
            elif norm_type == 'z-score':
                x_train, x_test, y_train, y_test = train_test_split(X_z_sm, y_small, test_size=0.2, random_state=42, stratify=y_small)
            elif norm_type == 'min-max':
                x_train, x_test, y_train, y_test = train_test_split(X_minmax_sm, y_small, test_size=0.2, random_state=42, stratify=y_small)

            accuracy = xgb_score(x_train, x_test, y_train, y_test, norm_type, r)
            results_dict_small[r][norm_type] = accuracy
    print(results_dict_small)
 
    ## Plotting for small dataset 

    ## Large Dataset
    for r in random_states:
        # # # Split the data
        results_dict_large[r] = {}
        for norm_type in normalization_types:
            if norm_type == 'unnormalized':
                x_train, x_test, y_train, y_test = train_test_split(X_large, y_large, test_size=0.2, random_state=42, stratify=y_large)
            elif norm_type == 'z-score':
                x_train, x_test, y_train, y_test = train_test_split(X_z_lg, y_large, test_size=0.2, random_state=42, stratify=y_large)
            elif norm_type == 'min-max':
                x_train, x_test, y_train, y_test = train_test_split(X_minmax_lg, y_large, test_size=0.2, random_state=42, stratify=y_large)

            accuracy = xgb_score(x_train, x_test, y_train, y_test, norm_type, r)
            results_dict_large[r][norm_type] = accuracy
    print(results_dict_large)

    Xg_bar_plots(results_dict_small, normalization_types, random_states)
    Xg_bar_plots(results_dict_large, normalization_types, random_states)



