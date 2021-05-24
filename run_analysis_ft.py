import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time

from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score, KFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_curve, auc, plot_roc_curve, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold

# define 10 by 3 nested crossvalidation
inner_fold = 3
outer_fold = 10

INPUT_DIR = "./../result/final_datasets_new_features/"
files = []
for file in os.listdir(INPUT_DIR):
    if 'csv' in file:
        files.append(file)

np.random.seed(1)

model_names = ["Logistic Regression", "SVM", "Random Forest", "Decision Tree"]

models = [
              Pipeline([('normalizer', StandardScaler()),  # normalize data
                      ('clf', LogisticRegression(random_state=1, max_iter=1000, class_weight="balanced"))  # fit Logistic regression model
              ]),    # pipeline for Logistic Regression
              Pipeline([('normalizer', StandardScaler()),  # normalize data
                  ('clf', SVC(random_state=1, class_weight="balanced"))  # fit SVM
              ]),    # pipeline for SVM
              RandomForestClassifier(random_state=1, class_weight="balanced"), # Random Forest
              DecisionTreeClassifier(random_state=1, class_weight="balanced")  # Decision Tree
         ]

params = [  # Hyperparameters need to be tuned 
            {
                'clf__solver' : ['newton-cg','lbfgs', 'liblinear'],
                'clf__penalty' : ["l2"],
                'clf__C' : [100, 10, 1.0, 0.1, 0.01]
            },
            {
                'clf__kernel' : ['poly', 'rbf', 'sigmoid'],
                'clf__gamma' : [10, 1, 0.1, 0.01, 0.001],
                'clf__C' : [100, 50, 10, 1.0, 0.1, 0.01, 0.001]
            },
            {
                'n_estimators' : [50, 100, 150, 200],
                'max_features' : ["auto", "sqrt", "log2"],
                'max_depth': [5, 10, 20, 30],
                'min_samples_leaf' : [1, 3, 5]
            },
            {
                'criterion' : ['gini', 'entropy'],
                'max_depth' : [5, 10, 20, 30, 40],
                'min_samples_leaf' : [1, 2, 3, 5, 10]
            }
          ]

colnames = ["model_name", "window_length", "accuracy", "f1-score", "precision", "recall", "auc"]
results = pd.DataFrame(columns=colnames)

num = 0
for file in files:
    num += 1
    print(num, " ", file)
    data = pd.read_csv("./../result/final_datasets_new_features/" + file)
    window_length = int(file.split(".")[0])
    groups = data.PID
    X = data.loc[:, ["mvm", "sdvm", "df", "p625", "fpdf", "mangle", "sdangle",
                     "mean_x", "mean_y", "mean_z",
                     "sd_x", "sd_y", "sd_z",
                     "cv_vm", "cv_x", "cv_y", "cv_z",
                     "min_vm", "min_x", "min_y", "min_z",
                     "max_vm", "max_x", "max_y", "max_z",
                     "lower_25_vm", "lower_25_x", "lower_25_y", "lower_25_z",
                     "upper_75_vm", "upper_75_x", "upper_75_y", "upper_75_z",
                     "third_moment_vm", "third_moment_x", "third_moment_y", "third_moment_z",
                     "fourth_moment_vm", "fourth_moment_x", "fourth_moment_y", "fourth_moment_z",
                     "skewness_vm", "skewness_x", "skewness_y", "skewness_z",
                     "kurtosis_vm", "kurtosis_x", "kurtosis_y", "kurtosis_z"]]
    y = data["group"]
    
    # nested cross-validation
    # Outer CV train-validation on 10 participant, test on the other 1
    # Inner CV 9 participant in total, train on 6, validate on 3
    # configure the cross-validation procedure
    inner_cv = GroupKFold(n_splits=inner_fold)
    outer_cv = GroupKFold(n_splits=outer_fold)
    
    for item in zip(model_names, models, params):
        name = item[0]
        model = item[1]
        param = item[2]
        
        acc = []
        f1 = []
        precision = []
        recall = []
        auc_score = []
        
        start = time.time()
        for train_index, test_index in outer_cv.split(X, y, groups=groups):
            x_train, x_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y[train_index], y[test_index]

            grid = GridSearchCV(estimator=model,
                                param_grid=param,
                                cv=inner_cv,
                                scoring="accuracy",
                                refit=True,
                                verbose=0,
                                n_jobs=-1)
    
            grid.fit(x_train, y_train, groups=groups[train_index])
            prediction = grid.predict(x_test)
            
            _acc = accuracy_score(y_test, prediction)
            _f1 = f1_score(y_test, prediction)
            _precision = precision_score(y_test, prediction)
            _recall = recall_score(y_test, prediction)
            _fpr, _tpr, _thresholds = roc_curve(y_test, prediction, pos_label=1)
            _auc = auc(_fpr, _tpr)
    
            acc.append(_acc)
            f1.append(_f1)
            precision.append(_precision)
            recall.append(_recall)
            auc_score.append(_auc)
        end = time.time()
        
        performance_dict = {
                "model_name" : [name],
                "window_length" : [window_length],
                "accuracy": [np.mean(acc)],
                "f1-score": [np.mean(f1)],
                "precision": [np.mean(precision)],
                "recall" : [np.mean(recall)],
                "auc" : [np.mean(auc_score)],
                "run_time" : [end-start]
                }
        
        performance = pd.DataFrame.from_dict(performance_dict)
        results = pd.concat([results, performance])

decision_tree = results.loc[results.model_name == "Decision Tree", :].copy()
linear_regression = results.loc[results.model_name == "Logistic Regression", :].copy()
SVM_clf = results.loc[results.model_name == "SVM", :].copy()
RF = results.loc[results.model_name == "Random Forest", :].copy()
decision_tree.sort_values(by = ["window_length"], inplace=True)
linear_regression.sort_values(by = ["window_length"], inplace=True)
SVM_clf.sort_values(by = ["window_length"], inplace=True)
RF.sort_values(by = ["window_length"], inplace=True)

# draw plot
plt.figure(figsize=(15,8))
plt.plot("window_length", "accuracy", data=decision_tree, marker='o', markerfacecolor='blue', markersize=10, color='skyblue', linewidth=2, label="Decision Tree")
plt.plot("window_length", "accuracy", data=linear_regression, marker='o', markerfacecolor='orange', markersize=10, color='orange', linewidth=2, label="Logistic Regression")
plt.plot("window_length", "accuracy", data=SVM_clf, marker='o', markerfacecolor="red", markersize=10, color='red', linewidth=2, label="SVM")
plt.plot("window_length", "accuracy", data=RF, marker='o', markerfacecolor='olive', markersize=10, color='olive', linewidth=2, label="Random Forest")
plt.legend()
plt.xlabel('Window-Length')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. Window-Length [FT Recognition]')
plt.ylim([0.65, 0.95])

