

import xgboost as xgb
import numpy as np
import time
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


"""
GPU Accelerated XGBoost

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

X.shape, y.shape
# ((581012, 54), (581012,))

print(f"target variable has {len(set(y))} unique values")
# target variable has 7 unique values

# Create train & test splits-
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    train_size = 0.75, test_size = 0.25,
    stratify = y
    )

X_train.shape, y_train.shape
# ((435759, 54), (435759,))

X_test.shape, y_test.shape
# ((145253, 54), (145253,))


# Specify number of boosting iterations to achieve a minima-
num_boosting_rounds = 3500

# Use default parameters-
params = {
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


# Convert dataset from np arrays to XGBoost specific formats-
dtrain = xgb.DMatrix(data = X_train, label = y_train)
dval = xgb.DMatrix(data = X_test, label = y_test)


# Python3 dict to contain accuracy results during training-
gpu_acc = {}
# cpu_acc = {}

start = time.time()

# Train XGBoost classifier model-
xgb_clf = xgb.train(
    params = params, dtrain = dtrain,
    num_boost_round = num_boosting_rounds,
    evals = [(dval, 'val')],
    evals_result = gpu_acc
    # evals_result = cpu_acc
    )

end = time.time()
print(f"XGBoost GPU training time = {end - start} seconds")
# XGBoost CPU training time = 432.7983458042145 seconds
# XGBoost GPU training time = 221.20399975776672 seconds


# Visualize training-
plt.plot(gpu_acc['val']['mlogloss'])
plt.xlabel("num boosting rounds")
plt.ylabel("mlogloss")
plt.title("XGBoost training Visualization")
plt.show()


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
# accuracy = 96.83%, precision = 95.04% & recall = 93.62%

print(f"Confusion matrix:\n{confusion_matrix(y_true = y_test, y_pred = y_pred)}")
'''
Confusion matrix:
[[51173  1654     1     0    21     4   107]
 [ 1373 69145    81     0   153    51    22]
 [    0    98  8646    35    12   148     0]
 [    0     0    59   595     0    33     0]
 [   16   259    15     0  2077     5     1]
 [    5    79   177    21     3  4057     0]
 [  137    27     0     0     1     0  4962]]
'''


# Most of the parameters are defaults. Consider hyper-parameter tuning to optimizie performance
# further.

