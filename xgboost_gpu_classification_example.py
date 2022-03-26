

import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import XGBClassifier, plot_importance
import time
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


"""
GPU Accelerated XGBoost:
To install GPU enabled xgboost package, the following command was used within an anaconda env-
pip install xgboost


Refer-
https://xgboost.readthedocs.io/en/latest/gpu/index.html
https://github.com/dmlc/xgboost/blob/master/demo/gpu_acceleration/cover_type.py
https://xgboost.readthedocs.io/en/stable/get_started.html
https://practicaldatascience.co.uk/machine-learning/how-to-tune-model-hyper-parameters-with-grid-search
"""


# This is an example code to demonstrate XGBoost's GPU usage . Hence, no data preprocessing, cleaning,
# etc. is performed.

# Get sklearn dataset-
X = fetch_covtype().data
y = fetch_covtype().target

# Sanity check-
X.shape, y.shape
# ((581012, 54), (581012,))

#Get target distribution information-
print(f"Number of unique target labels = {len(set(y))}")
# Number of unique target labels = 7

labels, dist = np.unique(y, return_counts = True)

for l, d in zip(labels, dist):
    print(f"label: {l} has {d} values")
'''
label: 1 has 211840 values
label: 2 has 283301 values
label: 3 has 35754 values
label: 4 has 2747 values
label: 5 has 9493 values
label: 6 has 17367 values
label: 7 has 20510 values
'''


# Create train and test splits-
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    train_size = 0.8, test_size = 0.2,
    stratify = y
)
    
# Sanity check-
X_train.shape, y_train.shape
# ((464809, 54), (464809,))

X_test.shape, y_test.shape
# ((116203, 54), (116203,))




# XGBoost classification:

# Specify number of boosting iterations to achieve a minima-
num_boosting_rounds = 3500

# To enable GPU training the only thing needed is to pass the ‘gpu_hist’ parameter-
param_dict = {
    # Use multi-class classification-
    'objective': 'multi:softmax',
    # Specify number of multi-classes-
    'num_class': 8,
    # Use GPU accelerated algorithm-
    'tree_method': 'gpu_hist',
    # Use CPU accelerated algorithm-
    # 'tree_method': 'hist',
    'eval_metric': 'mlogloss'
}

# NOTE:
# target variable has 7 unique values, but, 'num_class' = 8!

# Initialize an instance of classifier-
# xgb_clf = xgb.XGBClassifier(**param_dict)


# Convert dataset from np arrays to XGBoost specific formats-
dtrain = xgb.DMatrix(data = X_train, label = y_train)
dval = xgb.DMatrix(data = X_test, label = y_test)

# Python3 dict to contain accuracy results during training-
gpu_acc = {}
# cpu_acc = {}


start = time.time()

# Train XGBoost classifier model-
xgb_clf = xgb.train(
    params = param_dict, dtrain = dtrain,
    num_boost_round = num_boosting_rounds,
    evals = [(dval, 'val')],
    evals_result = gpu_acc
    # evals_result = cpu_acc
    )

end = time.time()
print(f"XGBoost GPU training time = {end - start:.2f} seconds")
# XGBoost GPU training time = 221.67 seconds


# Make predictions using trained model-
y_pred = xgb_clf.predict(dval)

# Compute metrics of trained model-
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average = 'macro')
rec = recall_score(y_test, y_pred, average = 'macro')

print("\nTrained (default) XGBoost's val metrics are:")
print(f"accuracy = {acc * 100:.2f}%, precision = {prec * 100:.2f}%"
f" & recall = {rec * 100:.2f}%")
# Trained (default) XGBoost's val metrics are:
# accuracy = 96.89%, precision = 95.06% & recall = 93.82%

print(f"Confusion matrix:\n{confusion_matrix(y_true = y_test, y_pred = y_pred)}")
'''
Confusion matrix:
[[40888  1359     2     0    11     3   105]
 [ 1029 55404    58     0   104    51    15]
 [    1    86  6917    35     7   105     0]
 [    0     0    50   477     0    22     0]
 [    9   178    22     0  1679    11     0]
 [    4    61   139    16     3  3250     0]
 [  111    17     0     0     0     0  3974]]
'''


# Visualize training: Validation data - logistic loss/cross-entropy loss
plt.plot(range(len(gpu_acc['val']['mlogloss'])), gpu_acc['val']['mlogloss'])
plt.xlabel("number of boosting rounds")
plt.ylabel("cross-entropy loss")
plt.title("XGBoost Training: Validation - cross-entropy loss")
plt.show()


# Visualize feature-wise importance-
plot_importance(xgb_clf)
plt.show()

print(f"XGBoost classifier booster: {xgb_clf.booster}")
# XGBoost classifier booster: gbtree

# There are 3500 trained tress. To see the 'optimal' number of trees-
xgb_clf.best_ntree_limit
# 3500

xgb_clf.best_iteration
# 3499


# Save trained XGBoost classifier to HDD-

# Save as JSON file-
xgb_clf.save_model("xgb_clf_gpu_trained.json")

# Save as text file-
# xgb_clf.save_model("xgb_clf_gpu_trained.txt")

# Load saved model from HDD using sklearn API-
# xgb_loaded.load_model("xgb_clf_gpu_trained.json")
'''
C:\Users\arjun\anaconda3\envs\traditional_ml\lib\site-packages\xgboost\sklearn.py:604: UserWarning: Loading a native XGBoost model with Scikit-Learn interface.
  warnings.warn(
'''

# Load save model using native XGBoost-
xgb_loaded = xgb.Booster()
xgb_loaded.load_model("xgb_clf_gpu_trained.json")

# Sanity check-
# xgb_loaded.best_ntree_limit
# 3500

# y_pred =  xgb_loaded.predict(data = dval)

