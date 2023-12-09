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
import xgboost 
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
import shap
from matplotlib_venn import venn3
from tqdm import tqdm
import json
import pickle

def filter_shap(test_data, shap_arr, y_map_new):
    df_data = []

    ids_list = test_data.index.to_list()
    genes_list = test_data.columns.to_list()
    genes_list = genes_list[0:19648]

    for i in tqdm(range(shap_arr.shape[0])):
        sample = shap_arr[i]
        sample_id = ids_list[i]

        label = y_map_new['pred'][i]

        w, h = sample.shape
        shap_scores_flat = sample[label][: len(genes_list)]
        df_data.append([sample_id, *list(shap_scores_flat), label])

    shap_df = pd.DataFrame(
        data = np.array(df_data), columns=["id", *genes_list, "predicted_label"]
    )

    shap_df.set_index("id", inplace=True)
    shap_df["true_label"] = list(y_map_new['true_label'])

    shap_df = pd.concat([shap_df], axis=1)

    return shap_df


def load_models(model_filenames):
    models = []
    for model_filename in model_filenames:
        with open(model_filename, 'rb') as pickle_file:
            models.append(pickle.load(pickle_file))
    return models




if __name__ == "__main__":
    print(os.getcwd())
    smallcsv = "Dataset_10k_1.csv"  # load large csv to memory
    smallpd = pd.read_csv(smallcsv, header=0)  # load small csv to memory
    targets = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # define 10 classes for single digits

    X_small = smallpd.drop(smallpd.columns[[0, 1]], axis=1)  # drop first 2 columns
    y_small = smallpd['0']  # these are the labels


    # use z-score normalization
    X_z_sm = stats.zscore(X_small)

    # user min-max normalization
    scaler = MinMaxScaler()
    X_minmax_sm = scaler.fit_transform(X_small)

    random_states = [0, 10, 20, 30, 40]
    normalization_types = ['unnormalized', 'z-score', 'min-max']
    results_dict_small = {}

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
            # Save split data
            df_x_test = pd.DataFrame(x_test)
            df_y_test = pd.DataFrame(y_test)
            df_x_test.to_csv(f"x{norm_type}_test{r}.csv", index=False)
            df_y_test.to_csv(f"y{norm_type}_test{r}.csv", index=False)


    #Shapley
    for i in normalization_types:
        for r in  range(len(random_states)):

            if i == "Unnormalized":
                name = "Unnormalized"
            elif i == "z-Score Normalization":
                name = "z-Score Normalization"
            else:
                name = "min-max Normalization"

            #Loading saved model
            # Load the pre-trained model
            xgbc = xgboost.Booster()
            with open(f'model_{i}_fold_{r}.pkl', 'rb') as pickle_file:
                xgbc = pickle.load(pickle_file)


            X_test = pd.read_csv(f'x{i}_test{r}.csv')

            Y_test = pd.read_csv(f'y{i}_test{r}.csv')
            Y_test = pd.DataFrame(Y_test)
            Y_test = Y_test.rename(columns = {'0': 'true_label'})

            explainer = shap.TreeExplainer(xgbc.get_booster())

            #Calculate SHAP score
            out_list = []
            num_samples = np.shape(X_test)[0]

            y_map_new = pd.DataFrame({'true_label': Y_test['true_label']})
            y_map_new['pred'] = xgbc.predict(X_test)

            for sample in tqdm(range(0,(num_samples))):
                shap_values = explainer.shap_values(X_test[sample: sample + 1])
                out_list.append(shap_values)

            shap_arr = np.squeeze(np.array(out_list))

            shap_df = filter_shap(X_test, shap_arr, y_map_new)


            top_20_features = shap_df.apply(lambda row: row.drop(['predicted_label', 'true_label']).abs().nlargest(20).index.tolist(), axis=1)

            for k in range(10):
                #Create Venn Diagrams
                labelSamples = top_20_features[shap_df['true_label'] == k].sample(3, random_state=42)


                venn3([set(labelSamples.iloc[0]),
                       set(labelSamples.iloc[1]),
                       set(labelSamples.iloc[2])],
                    set_labels=('Sample 1', 'Sample 2', 'Sample 3'))

                plt.title(f"Top 20 Features {name} fold {r} | class {k}")

                fileName = f'vennDiagram{i}{r}{k}.png'
                plt.savefig(fileName)

                plt.show()
                plt.close()
                plt.figure()