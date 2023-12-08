import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, KFold
# import class for z-score normalization
import scipy.stats as stats
# import class for min-max scaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from matplotlib_venn import venn3
import shap
import collections
from sklearn.model_selection import train_test_split


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

    # Set plot labels and title
    plt.xlabel('MNIST class')
    plt.ylabel('Value')
    plt.title('Violin Plot for ' + n)

    # Set x-axis tick labels
    plt.xticks(np.arange(1, len(t) + 1), t)

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


# filtering samples based on true predicted label
def filter_shap(test_data, shap_array, y_true, y_map_new):
    df_data = []
    ids_list = list(range(0, len(test_data) - 1))
    pix_list = list(range(0, 784))
    # pix_list = pix_list[0: 783]

    for i, sampl in enumerate(shap_array[0]):
        # print(sampl)
        sample_id = ids_list[i]
        label = y_map_new[i]
        truelabel = np.array(y_true)[i]
        # print("label :", label)
        # label = pl.Expr.map_dict[label]
        # print(label)
        shap_scores_flat = sampl[: len(pix_list)]

        df_data.append([sample_id, *list(shap_scores_flat), label, truelabel])

    shap_df = pd.DataFrame(data=np.array(df_data), columns=['id', *pix_list, 'predicted_label', 'true_label'])
    shap_df.set_index('id', inplace=True)
    # shap_df['true_label'] = y_true

    return shap_df


if __name__ == "__main__":
    largepd = pd.read_csv(largecsv, header=0)  # load large csv to memory
    smallpd = pd.read_csv(smallcsv, header=0)  # load small csv to memory
    # largepd = pd.read_csv("Combined_MNIST_70k.csv", header=0)  # load large csv to memory
    # smallpd = pd.read_csv("Dataset_10k_1.csv", header=0)  # load small csv to memory

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

    # +++++++++++++++++++++++++++++++
    # ++++++++   OUTPUT 1    +++++++
    # +++++++++++++++++++++++++++++++
    # get the counts
    counterlarge = collections.Counter(y_large)
    print(counterlarge)
    countersmall = collections.Counter(y_small)
    print(countersmall)

    # +++++++++++++++++++++++++++++++
    # ++++++++   OUTPUT 2    +++++++
    # +++++++++++++++++++++++++++++++
    # draw violin plots with not-normalized data
    draw_violin(X_small_mean, y_small, targets, 'Small dataset not-normalized')
    draw_violin(X_large_mean, y_large, targets, 'Large dataset not-normalized')

    # violin plots with z-score normalized data
    draw_violin(X_z_sm_mn, y_small, targets, 'Small dataset z-score')
    draw_violin(X_z_lg_mn, y_large, targets, 'Large dataset z-score')

    # violin plots with min-max normalized data
    draw_violin(X_minmax_sm_mn, y_small, targets, 'Small dataset min-max')
    draw_violin(X_minmax_lg_mn, y_large, targets, 'Large dataset min-max')

    # +++++++++++++++++++++++++++++++
    # ++++++++   OUTPUT 3    +++++++
    # +++++++++++++++++++++++++++++++
    # Experimental design for XGBoost
    # set the seed
    seedloop = [0, 10, 20, 30, 40]
    xg_accuracies_sm = []
    xg_accuracies_sm_z = []
    xg_accuracies_sm_m = []
    xg_accuracies_lg = []
    xg_accuracies_lg_z = []
    xg_accuracies_lg_m = []

    # not not-normalized data
    # split data set for train and test with 80/20 stratified split
    X_train_lg, X_test_lg, y_train_lg, y_test_lg = train_test_split(X_large.to_numpy(), y_large, stratify=y_large,
                                                                    test_size=0.20, random_state=0)
    X_train_sm, X_test_sm, y_train_sm, y_test_sm = train_test_split(X_small.to_numpy(), y_small, stratify=y_small,
                                                                    test_size=0.20, random_state=0)

    # split data set after normalizing with z-score
    X_trainz_lg, X_testz_lg, y_trainz_lg, y_testz_lg = train_test_split(X_z_lg, y_large, stratify=y_large,
                                                                        test_size=0.20, random_state=0)
    X_trainz_sm, X_testz_sm, y_trainz_sm, y_testz_sm = train_test_split(X_z_sm, y_small, stratify=y_small,
                                                                        test_size=0.20, random_state=0)
    # split data set after normalizing with min-max
    X_trainm_lg, X_testm_lg, y_trainm_lg, y_testm_lg = train_test_split(X_minmax_lg, y_large, stratify=y_large,
                                                                        test_size=0.20, random_state=0)
    X_trainm_sm, X_testm_sm, y_trainm_sm, y_testm_sm = train_test_split(X_minmax_sm, y_small, stratify=y_small,
                                                                        test_size=0.20, random_state=0)

    # loop for seed values large dataset
    for i in seedloop:
        xg_accuracies_lg.append(xgb_score(X_train_lg, y_train_lg, X_test_lg, y_test_lg, i))
        xg_accuracies_lg_z.append(xgb_score(X_trainz_lg, y_trainz_lg, X_testz_lg, y_testz_lg, i))
        xg_accuracies_lg_m.append(xgb_score(X_trainm_lg, y_trainm_lg, X_testm_lg, y_testm_lg, i))

    print("XGBoost Accuracies LG not-normalized - ", xg_accuracies_lg)
    print("XGBoost Accuracies LG zscore - ", xg_accuracies_lg_z)
    print("XGBoost Accuracies LG minmax - ", xg_accuracies_lg_m)

    # draw plot with classifier results
    plotdata = pd.DataFrame({
        "All genes not normalized": xg_accuracies_lg,
        "All genes z-score": xg_accuracies_lg_z,
        "All genes min-max": xg_accuracies_lg_m
    },
        index=["0", "10", "20", "30", "40"]
    )
    plotdata.plot(kind="bar", figsize=(15, 8))

    plt.title("XGBoost Large Dataset")

    plt.xlabel("Random State")

    plt.ylabel("Accuracy")

    plt.legend(prop={'size': 10}, loc='lower right')
    plt.show()

    # loop for seed values small dataset
    for i in seedloop:
        xg_accuracies_sm.append(xgb_score(X_train_sm, y_train_sm, X_test_sm, y_test_sm, i))
        xg_accuracies_sm_z.append(xgb_score(X_trainz_sm, y_trainz_sm, X_testz_sm, y_testz_sm, i))
        xg_accuracies_sm_m.append(xgb_score(X_trainm_sm, y_trainm_sm, X_testm_sm, y_testm_sm, i))

    print("XGBoost Accuracies SM not-normalized - ", xg_accuracies_sm)
    print("XGBoost Accuracies SM zscore - ", xg_accuracies_sm_z)
    print("XGBoost Accuracies SM minmax - ", xg_accuracies_sm_m)

    # draw plot with classifier results
    plotdata = pd.DataFrame({
        "All genes not normalized": xg_accuracies_sm,
        "All genes z-score": xg_accuracies_sm_z,
        "All genes min-max": xg_accuracies_sm_m
    },
        index=["0", "10", "20", "30", "40"]
    )
    plotdata.plot(kind="bar", figsize=(15, 8))

    plt.title("XGBoost")

    plt.xlabel("Random State")

    plt.ylabel("Accuracy")

    plt.legend(prop={'size': 10}, loc='lower right')
    plt.show()

    # +++++++++++++++++++++++++++++++++++++++++++++++
    # ++++++++++++++++   SHAP    Output 4
    # +++++++++++++++++++++++++++++++++++++++++++++++

    # Data distribution
    class_distribution = {
        0: 986,
        1: 1126,
        2: 998,
        3: 1020,
        4: 975,
        5: 902,
        6: 983,
        7: 1041,
        8: 975,
        9: 994
    }



    # Preprocess data
    X_large = largepd.drop(largepd.columns[[0, 1]], axis=1)  # drop first 2 columns
    y_large = largepd['0']  # these are the labels
    X_small = smallpd.drop(smallpd.columns[[0, 1]], axis=1)  # drop first 2 columns
    y_small = smallpd['0']  # these are the labels

    # Split small dataset into train and test
    X_small_train, X_small_test, y_small_train, y_small_test = train_test_split(X_small, y_small, test_size=0.2, random_state=42)

    # Initialize XGBoost classifier
    xg_clf = XGBClassifier(learning_rate=0.3, n_estimators=150, max_depth=6, min_child_weight=1, gamma=0,
                            reg_lambda=1, subsample=1, colsample_bytree=1, objective='multi:softprob', num_class=10, random_state=10)

    # Train the model on the small dataset
    xg_clf.fit(X_small_train, y_small_train)

    # Initialize SHAP explainer
    explainer = shap.TreeExplainer(xg_clf)


    # Function to create and save Local Feature Importance DataFrames
    def create_and_save_feature_importance_df(X, normalization_type, save_path):
        shap_values = explainer.shap_values(X)
        
        # For multi-class classification, take SHAP values for the correct class
        if len(shap_values) > 1:
            shap_values = shap_values[y_small.unique().tolist().index(0)]  # Assuming class 0, adjust as needed
        
        feature_importance_df = pd.DataFrame(shap_values, columns=X.columns)
        
        # Save DataFrame to CSV
        print(feature_importance_df)
        feature_importance_df.to_csv(save_path, index=False)
        print(f"Local Feature Importance DataFrame: {normalization_type} created and saved at {save_path}")




    # Create and save Local Feature Importance DataFrames
    create_and_save_feature_importance_df(X_small, "unnormalized", "Local_Feature_Importance_DataFrame/Local_Feature_Importance_unnormalized.csv")
    create_and_save_feature_importance_df(X_small_train, "z-score", "Local_Feature_Importance_DataFrame/Local_Feature_Importance_z-score.csv")
    create_and_save_feature_importance_df(X_small_train, "min-max", "Local_Feature_Importance_DataFrame/Local_Feature_Importance_min-max.csv")
    
    
    #++++++++++++++++ For venn diagrams with 5 fold 5-fold cross-validation

   
    # Initialize SHAP explainer
    explainer = None

    # Normalization types
    normalization_types = ["unnormalized", "z-score", "min-max"]

    # Perform 5-fold cross-validation for each normalization type
    for normalization_type in normalization_types:
        # Preprocess the data based on normalization type
        if normalization_type == "z-score":
            scaler = StandardScaler()
        elif normalization_type == "min-max":
            scaler = MinMaxScaler()
        else:
            scaler = None

        # Initialize cross-validation
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=10)

        for train_index, test_index in kf.split(X_small, y_small):
            X_train_fold, X_test_fold = X_small.iloc[train_index], X_small.iloc[test_index]
            y_train_fold, y_test_fold = y_small.iloc[train_index], y_small.iloc[test_index]

            # Apply normalization
            if scaler is not None:
                X_train_fold = scaler.fit_transform(X_train_fold)
                X_test_fold = scaler.transform(X_test_fold)

            # Train XGBoost classifier
            model = XGBClassifier(xg_clf)
            model.fit(X_train_fold, y_train_fold)

            # Create SHAP explainer if not created yet
            if explainer is None:
                explainer = shap.TreeExplainer(model)

            # Apply SHAP explainer for each fold
            shap_values = explainer.shap_values(X_test_fold)

            # Convert shap_values list to NumPy array
            shap_values = np.array(shap_values)

            # Isolate top 20 features for each sample
            top_features_indices = np.argsort(np.abs(shap_values), axis=2)[:, :, -20:]

            # Reshape top_features_indices to 2D array
            top_features_indices_2d = top_features_indices.reshape(-1, 20)

            # Comparing Samples in a Class
            num_classes = len(np.unique(y_test_fold))
            for class_label in range(num_classes):
                # Randomly select three samples for each class
                samples_in_class = X_test_fold[y_test_fold == class_label]
                samples_in_class_df = pd.DataFrame(samples_in_class)
                selected_samples = samples_in_class_df.sample(min(3, len(samples_in_class)), random_state=10)

                # Ensure selected_samples.index is within the bounds of top_features_indices_2d
                valid_indices = selected_samples.index[selected_samples.index < top_features_indices_2d.shape[0]]

                # Draw Venn diagram using the top 20 features for each set of three samples
                feature_sets = [set(top_features_indices_2d[sample_id]) for sample_id in valid_indices]
                venn3(feature_sets, set_labels=[f"Sample {i+1}" for i in range(len(feature_sets))])
                plt.title(f"Class {class_label} - Top 20 Features Intersection ({normalization_type})")
                plt.savefig(f"VennDiagrams/Class_{class_label}_{normalization_type}_trainindex:{train_index}_testindex{test_index}Venn.png")
                plt.close() 

    # +++++++++++++++++++++++++++++++
    # ++++++++   OUTPUT 5    +++++++
    # +++++++++++++++++++++++++++++++

    from sklearn.model_selection import train_test_split

    xg_clf = XGBClassifier(learning_rate=0.3, n_estimators=150, max_depth=6, min_child_weight=1, gamma=0,
                           reg_lambda=1, subsample=1, colsample_bytree=1, objective='multi:softprob', num_class=10,
                           random_state=10)

    model = XGBClassifier(xg_clf)

    X = X_small
    y = y_small

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the XGBoost model

    model.fit(X_train, y_train)

    # Predict on the test set
    predictions = model.predict(X_test)
    # Identify correctly predicted samples
    correctly_predicted = predictions == y_test

    # Compute SHAP values for the test set
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test)

    # Convert the boolean series to a numpy array for proper indexing
    correctly_predicted_np = correctly_predicted.to_numpy()
    # Process for each class

    for cls in np.unique(y_test):
        # Filter SHAP values for correctly predicted samples of the class
        class_indices = (y_test == cls).to_numpy() & correctly_predicted_np
        class_shap_values = shap_values[class_indices]
        # Flatten the SHAP values across all dimensions except for features.
        # Ensure that the result is a one-dimensional array
        flattened_shap_values = np.abs(class_shap_values.values).reshape(-1, X_test.shape[1]).mean(axis=0)
        # Extract top 20 features
        top_features_indices = np.argsort(flattened_shap_values)[::-1][:20]
        top_features = X_test.columns[top_features_indices]
        top_shap_values = flattened_shap_values[top_features_indices]
        # Draw barplot for top 20 features
        plt.figure(figsize=(10, 6))
        plt.barh(range(20), top_shap_values, tick_label=top_features)
        plt.gca().invert_yaxis()
        plt.title(f"Top 20 Features for Class {cls} Based on Mean SHAP Values")
        plt.xlabel("Mean Absolute SHAP Value")
        plt.show()













#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


#     xg_clf = XGBClassifier(learning_rate=0.3, n_estimators=150, max_depth=6, min_child_weight=1, gamma=0, reg_lambda=1, subsample=1, colsample_bytree=1, objective='multi:softprob', num_class=10, random_state=10)
#     # Initialize SHAP explainer
#     explainer = None

#     # Normalization types
#     normalization_types = ["unnormalized", "z-score", "min-max"]

#     # Perform 5-fold cross-validation for each normalization type
#     for normalization_type in normalization_types:
#         # Preprocess the data based on normalization type
#         if normalization_type == "z-score":
#             scaler = StandardScaler()
#         elif normalization_type == "min-max":
#             scaler = MinMaxScaler()
#         else:
#             scaler = None

#         # Initialize cross-validation
#         kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=10)

#         for train_index, test_index in kf.split(X_small, y_small):
#             X_train_fold, X_test_fold = X_small.iloc[train_index], X_small.iloc[test_index]
#             y_train_fold, y_test_fold = y_small.iloc[train_index], y_small.iloc[test_index]

#             # Apply normalization
#             if scaler is not None:
#                 X_train_fold = scaler.fit_transform(X_train_fold)
#                 X_test_fold = scaler.transform(X_test_fold)

#             # Train XGBoost classifier
#             model = XGBClassifier(xg_clf)
#             model.fit(X_train_fold, y_train_fold)

#             # Create SHAP explainer if not created yet
#             if explainer is None:
#                 explainer = shap.TreeExplainer(model)

                
#             # Apply SHAP explainer for each fold
#             shap_values = explainer.shap_values(X_test_fold)

#             # Print the shape of shap_values
#             print("Shape of shap_values:", np.array(shap_values).shape)

#             # Convert shap_values list to NumPy array
#             shap_values = np.array(shap_values)

#             # Print the updated shape of shap_values
#             print("Updated shape of shap_values:", shap_values.shape)

#             # Isolate top 20 features for each sample
#             top_features_indices = np.argsort(np.abs(shap_values), axis=2)[:, :, -20:]

#             # Print the shape of top_features_indices
#             print("Shape of top_features_indices:", top_features_indices.shape)

#             # Reshape top_features_indices to 2D array
#             top_features_indices_2d = top_features_indices.reshape(-1, 20)


#             # Create DataFrame for local feature importance
#             temp_df = pd.DataFrame(index=np.arange(len(X_test_fold) * 10))
#             shap_df_local = pd.DataFrame(top_features_indices_2d, index=temp_df.index,
#                                         columns=[f"Feature_{i}" for i in range(1, 21)])


#             # Comparing Samples in a Class
#             num_classes = len(np.unique(y_test_fold))
#             for class_label in range(num_classes):
#                 # Randomly select three samples for each class
#                 samples_in_class = X_test_fold[y_test_fold == class_label]
#                 samples_in_class_df = pd.DataFrame(samples_in_class)
#                 selected_samples = samples_in_class.sample(min(3, len(samples_in_class)), random_state=10)

#                 # Ensure selected_samples.index is within the bounds of top_features_indices_2d
#                 valid_indices = selected_samples.index[selected_samples.index < top_features_indices_2d.shape[0]]

#                 # Draw Venn diagram using the top 20 features for each set of three samples
#                 feature_sets = [set(top_features_indices_2d[sample_id]) for sample_id in valid_indices]
#                 venn3(feature_sets, set_labels=[f"Sample {i+1}" for i in range(len(feature_sets))])
#                 plt.title(f"Class {class_label} - Top 20 Features Intersection ({normalization_type})")
#                 plt.show()

#             # Output 4: Local Feature Importance
#             print("Local Feature Importance DataFrame:")
#             print(shap_df_local)

# #+++++++++++++++++++++++++++++++++++++++++++++++
# #+++++++++++++++++++++++++++++++++++++++++++++++
# # Adriana's code is bellow


#     # kf = KFold(n_splits=5)
#     # for k, (train_index, test_index) in enumerate(kf.split(X_small)):
#     #     X_train_sm = np.take(X_small.to_numpy(), train_index, axis=0)
#     #     # print("X train: ", X_train_sm.shape)
#     #     y_train_sm = np.take(y_small, train_index, axis=0)
#     #     # print("Y train: ", y_train_sm.shape)
#     #     X_test_sm = np.take(X_small.to_numpy(), test_index, axis=0)
#     #     # print("X test: ", X_test_sm.shape)
#     #     y_test_sm = np.take(y_small, test_index, axis=0)
#     #     # print("Y test: ", y_test_sm.shape)

#     #     csvfn = os.getcwd() + "\\shap_df" + str(k) + ".csv"
#     #     # print(csvfn)

#     #     xg_clf = XGBClassifier(learning_rate=0.3, n_estimators=150, max_depth=6, min_child_weight=1, gamma=0, reg_lambda=1,
#     #                            subsample=1, colsample_bytree=1, objective='multi:softprob', num_class=10, random_state=10)
#     #     xg_clf.fit(X_train_sm, y_train_sm)
#     #     y_predict_sm = xg_clf.predict(X_test_sm)
#     #     # print(y_predict_sm.shape)
#     #     # print(y_predict_sm)
#     #     # exit(0)
#     #     # explain the xgb model with SHAP
#     #     explainer_xgb = shap.TreeExplainer(xg_clf)
#     #     # calculating shap value
#     #     out_list = []
#     #     num_samples = np.shape(X_test_sm)[0]
#     #     for sample in tqdm(range(0, num_samples)):
#     #         # shap
#     #         shap_values = explainer_xgb.shap_values(X_test_sm[sample: sample + 1])
#     #         out_list.append(shap_values)

#     #     # squeeze this shap value for the test data
#     #     shap_arr = np.squeeze(np.array(out_list))
#     #     # print(shap_arr.shape[0])

#     #     shap_df = filter_shap(X_test_sm, shap_arr, y_test_sm, y_predict_sm)
#     #     # shap.plots.bar(shap_df1)
#     #     shap_df.to_csv(csvfn)


