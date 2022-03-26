

import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


"""
LightGBM GPU classification - toy example.


Reference-
https://github.com/microsoft/LightGBM/blob/master/examples/python-guide/advanced_example.py
https://lightgbm.readthedocs.io/en/latest/Python-API.html#training-api
"""


# Get sample training data-
train, label = make_moons(
    n_samples = 300000, shuffle = True,
    noise = 0.3, random_state = None
)

train.shape, label.shape
# ((300000, 2), (300000,))

print(f"number of unique labels in dataset = {len(set(label))}")
# number of unique labels in dataset = 2

# Get distribution of target labels-
lbl, dist = np.unique(label, return_counts = True)

for l, d in zip(lbl, dist):
    print(f"label: {l} has {d} values")
'''
label: 0 has 150000 values
label: 1 has 150000 values
'''

# Create train and test splits-
X_train, X_test, y_train, y_test = train_test_split(
    train, label,
    train_size = 0.8, test_size = 0.2,
    stratify = label
)
    
# Sanity check-
X_train.shape, y_train.shape
# ((240000, 2), (240000,))

X_test.shape, y_test.shape
# ((60000, 2), (60000,))


# Create lightgbm specific dataset-
# NOTE: if you want to re-use data, remember to set free_raw_data=False
lgb_train = lgb.Dataset(X_train, y_train, free_raw_data = False)
lgb_val = lgb.Dataset(X_test, y_test, free_raw_data = False)


# Specify configurations/hyper-parameters as Python3 dict-
params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'binary_logloss',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}

# Generate feature names
feature_name = [f'feature_{col}' for col in range(X_train.shape[1])]

feature_name
# ['feature_0', 'feature_1']


# Train LightGBM classifier-
lgb_clf = lgb.train(
    params = params, train_set = lgb_train,
    valid_sets = lgb_val, feature_name = feature_name
)

# Makre predictions using trained model-
y_pred = lgb_clf.predict(X_test)
# Can only predict with the best iteration (or the saving iteration)

# Convert to 0/1 from scores-
y_pred = np.where(y_pred < 0.5, 0, 1)

# Compute validation metrics-
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average = 'macro')
rec = recall_score(y_test, y_pred, average = 'macro')

print("Trained LightGBM classifier validation metrics")
print(f"accuracy = {acc * 100:.2f}%, precision = {prec * 100:.2f}% & recall = {rec * 100:.2f}%")
# Trained LightGBM classifier validation metrics
# accuracy = 91.42%, precision = 91.42% & recall = 91.42%

print(f"Confusion matrix:\n{confusion_matrix(y_true = y_test, y_pred = y_pred)}")
'''
Confusion matrix:
[[27434  2566]
 [ 2580 27420]]
'''


"""
# Initialize an instance of LGB classifier-
lgb_clf = LGBMClassifier(
    boosting_type = 'gbdt', num_leaves = 31,
    max_depth = -1, learning_rate = 0.1,
    n_estimators = 300, device = "gpu"
)

# Train model on training data-
lgb_clf.fit(train, label)
"""

"""
# Visualize feature importances-

# Save learned feature importance as Python3 dict-
for f, i in zip(lgb_clf.feature_name(), lgb_clf.feature_importance()):
    feature_importances_d[f] = i

# Sanity check-
feature_importances_d
# {'Column_0': 4857, 'Column_1': 4143}

# Visualize feature importance as a bar graph-
n = list(feature_importances_d.keys())
v = list(feature_importances_d.values())

plt.bar(range(len(feature_importances_d)), v, tick_label = n)
plt.xlabel("features")
plt.ylabel("feature importance scores")
plt.title("LightGBM: feature importance")
plt.show()

del n, v
"""

# Visualize feature importances-
lgb.plot_importance(booster = lgb_clf)
plt.title("LightGBM classifier: feature importance")
plt.show()

