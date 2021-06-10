

"""
Configure XGBoost for Imbalanced Classification


It is an efficient implementation of the stochastic gradient boosting algorithm and offers a range of
hyperparameters that give fine-grained control over the model training procedure. Although the algorithm
performs well in general, even on imbalanced classification datasets, it offers a way to tune the training
algorithm to pay more attention to misclassification of the minority class for datasets with a skewed
class distribution.

This modified version of XGBoost is referred to as Class Weighted XGBoost or Cost-Sensitive XGBoost and
can offer better performance on binary classification problems with a severe class imbalance.


Refer-
https://machinelearningmastery.com/xgboost-for-imbalanced-classification/
"""
from sklearn.datasets import make_classification
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import numpy as np
from scipy.stats import uniform as sp_uniform
from scipy.stats import randint as sp_randint
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter


# Define synthetic dataset-
X, y = make_classification(
    n_samples = 10000, n_features = 2,
    n_redundant = 0, n_clusters_per_class = 2,
    weights = [0.99], flip_y= 0
    )

X.shape, y.shape
# ((10000, 2), (10000,))

# Count unique values-
vals, count = np.unique(y, return_counts=True)

vals
# array([0, 1])

# count
# array([9900,  100], dtype=int64)

# Summarize class distribution-
counter = Counter(y)
# count examples in each class

counter
# Counter({0: 9900, 1: 100})

# We can see that the dataset has an approximate 1:100 class distribution with a little less than
# 10,000 examples in the majority class and 100 in the minority class.


# Visualize dataset-
sns.scatterplot(x = X[:, 0], y = X[:, 1], hue = y)
plt.show()




"""
XGBoost Model for Classification

XGBoost is short for Extreme Gradient Boosting and is an efficient implementation of the stochastic 
gradient boosting machine learning algorithm.

The stochastic gradient boosting algorithm, also called gradient boosting machines or tree boosting,
is a powerful machine learning technique that performs well or even best on a wide range of
challenging machine learning problems.

'Tree boosting has been shown to give state-of-the-art results on many standard classification
benchmarks.'
— XGBoost: A Scalable Tree Boosting System, 2016.

It is an ensemble of decision trees algorithm where new trees fix errors of those trees that are
already part of the model. Trees are added until no further improvements can be made to the model.

XGBoost provides a highly efficient implementation of the stochastic gradient boosting algorithm
and access to a suite of model hyperparameters designed to provide control over the model training process.

'The most important factor behind the success of XGBoost is its scalability in all scenarios. The
system runs more than ten times faster than existing popular solutions on a single machine and scales
to billions of examples in distributed or memory-limited settings.'
— XGBoost: A Scalable Tree Boosting System, 2016.

XGBoost is an effective machine learning model, even on datasets where the class distribution is skewed.

Before any modification or tuning is made to the XGBoost algorithm for imbalanced classification, it is
important to test the default XGBoost model and establish a baseline in performance.

Although the XGBoost library has its own Python API, we can use XGBoost models with the scikit-learn API
via the XGBClassifier wrapper class. An instance of the model can be instantiated and used just like any
other scikit-learn class for model evaluation.
"""
# Define model-
xgb_clf = xgb.XGBClassifier(
    objective = 'binary:logistic', eval_metric  = 'logloss',
    use_label_encoder = False)


# We will use repeated cross-validation to evaluate the model, with three repeats of 10-fold
# cross-validation.
# The model performance will be reported using the mean ROC area under curve (ROC AUC) averaged over
# repeats and all folds.
# Define evaluation procedure-
kfold_cv = RepeatedStratifiedKFold(n_splits = 10, n_repeats = 3)


# Evaluate model-
roc_scores = cross_val_score(
    estimator = xgb_clf, X = X,
    y = y, scoring = 'roc_auc',
    cv = kfold_cv)

acc_scores = cross_val_score(
    estimator = xgb_clf, X = X,
    y = y, scoring = 'accuracy',
    cv = kfold_cv)

prec_scores = cross_val_score(
    estimator = xgb_clf, X = X,
    y = y, scoring = 'precision',
    cv = kfold_cv)

rec_scores = cross_val_score(
    estimator = xgb_clf, X = X,
    y = y, scoring = 'recall',
    cv = kfold_cv)



print(f"Mean ROC-AUC with default XGBoost classifier = {np.mean(roc_scores):.4f}")
print(f"Mean accuracy with default XGBoost classifier = {np.mean(acc_scores) * 100:.3f}%")
print(f"Mean precision with default XGBoost classifier = {np.mean(prec_scores) * 100:.3f}%")
print(f"Mean recall with default XGBoost classifier = {np.mean(rec_scores) * 100:.3f}%")
# Mean ROC-AUC with default XGBoost classifier = 0.9755
# Mean accuracy with default XGBoost classifier = 99.687%
# Mean precision with default XGBoost classifier = 92.970%
# Mean recall with default XGBoost classifier = 75.667%

# We can see that the model has skill, achieving a ROC AUC above 0.5, in this case achieving a mean
# score of 0.9755.


# Compute confusion matrix-
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, stratify = y)

X_train.shape, y_train.shape, X_test.shape, y_test.shape
# ((7000, 2), (7000,), (3000, 2), (3000,))

# Sanity check - whether the ratio of majority & minority classes are preserved-
Counter(y_train)
# Counter({0: 6930, 1: 70})

Counter(y_test)
# Counter({0: 2970, 1: 30})

70 / 6930, 30 / 2970
# (0.010101010101010102, 0.010101010101010102)

# Train and predict-
xgb_clf.fit(X_train, y_train)
y_pred = xgb_clf.predict(X_test)

confusion_matrix(y_true = y_test, y_pred = y_pred)
'''
array([[2969,    1],
       [   8,   22]], dtype=int64)
'''




"""
Weighted XGBoost for Class Imbalance

Although the XGBoost algorithm performs well for a wide range of challenging problems, it offers a
large number of hyperparameters, many of which require tuning in order to get the most out of the
algorithm on a given dataset.

The implementation provides a hyperparameter designed to tune the behavior of the algorithm for
imbalanced classification problems; this is the 'scale_pos_weight' hyperparameter.

By default, the 'scale_pos_weight' hyperparameter is set to the value of 1.0 and has the effect of
weighing the balance of positive examples, relative to negative examples when boosting decision trees.
For an imbalanced binary classification dataset, the negative class refers to the majority class
(class 0) and the positive class refers to the minority class (class 1).

XGBoost is trained to minimize a loss function and the 'gradient' in gradient boosting refers to
the steepness of this loss function, e.g. the amount of error. A small gradient means a small error
and, in turn, a small change to the model to correct the error. A large error gradient during training
in turn results in a large correction.

- Small Gradient: Small error or correction to the model.
- Large Gradient: Large error or correction to the model.

Gradients are used as the basis for fitting subsequent trees added to boost or correct errors made by
the existing state of the ensemble of decision trees.

The 'scale_pos_weight' value is used to scale the gradient for the positive class.

This has the effect of scaling errors made by the model during training on the positive class and
encourages the model to over-correct them. In turn, this can help the model achieve better performance
when making predictions on the positive class. Pushed too far, it may result in the model overfitting
the positive class at the cost of worse performance on the negative class or both classes.

As such, the 'scale_pos_weight' can be used to train a class-weighted or cost-sensitive version of
XGBoost for imbalanced classification.

A sensible default value to set for the 'scale_pos_weight' hyperparameter is the inverse of the
class distribution. For example, for a dataset with a 1 to 100 ratio for examples in the minority to
majority classes, the 'scale_pos_weight' can be set to 100. This will give classification errors made
by the model on the minority class (positive class) 100 times more impact, and in turn, 100 times more
correction than errors made on the majority class.
"""

'''
The XGBoost documentation suggests a fast way to estimate this value using the training dataset as
the total number of examples in the majority class divided by the total number of examples in the
minority class, or,

- scale_pos_weight = # of examples in majority class / # of examples in minority class

- scale_pos_weight = total_negative_examples / total_positive_examples

For example, we can calculate this value for our synthetic classification dataset. We would expect this
to be about 100, or more precisely, 99 given the weighing we used to define the dataset.
'''
# estimate scale_pos_weight value-
estimate = counter[0] / counter[1]
print(f"scale_pos_weight estimate = {estimate:.4f}")
# scale_pos_weight estimate = 99.0000


# Define model-
xgb_clf = xgb.XGBClassifier(
    objective = 'binary:logistic', eval_metric  = 'logloss',
    use_label_encoder = False, scale_pos_weight = 99)

# Define evaluation procedure-
kfold_cv = RepeatedStratifiedKFold(n_splits = 10, n_repeats = 3)


# Evaluate model-
roc_scores = cross_val_score(
    estimator = xgb_clf, X = X,
    y = y, scoring = 'roc_auc',
    cv = kfold_cv)

acc_scores = cross_val_score(
    estimator = xgb_clf, X = X,
    y = y, scoring = 'accuracy',
    cv = kfold_cv)

prec_scores = cross_val_score(
    estimator = xgb_clf, X = X,
    y = y, scoring = 'precision',
    cv = kfold_cv)

rec_scores = cross_val_score(
    estimator = xgb_clf, X = X,
    y = y, scoring = 'recall',
    cv = kfold_cv)



print(f"Mean ROC-AUC with class-weighted XGBoost classifier = {np.mean(roc_scores):.4f}")
print(f"Mean accuracy with class-weighted XGBoost classifier = {np.mean(acc_scores) * 100:.3f}%")
print(f"Mean precision with class-weighted XGBoost classifier = {np.mean(prec_scores) * 100:.3f}%")
print(f"Mean recall with class-weighted XGBoost classifier = {np.mean(rec_scores) * 100:.3f}%")
# Mean ROC-AUC with class-weighted XGBoost classifier = 0.9630
# Mean accuracy with class-weighted XGBoost classifier = 99.117%
# Mean precision with class-weighted XGBoost classifier = 63.632%
# Mean recall with class-weighted XGBoost classifier = 57.000%

# Compute confusion matrix-
xgb_clf.fit(X_train, y_train)
y_pred = xgb_clf.predict(X_test)

confusion_matrix(y_true = y_test, y_pred = y_pred)
'''
array([[2968,    2],
       [   8,   22]], dtype=int64)
'''

# Observation: The class-weighted XGBoost classifier doesn't improve it's performance on the
# minority class (compare default vs. class-weighted confusion matrices). However, due to
# scale_pos_weight parameter, it misclassifies the majority class by 1 which changes model
# metrics




"""
Hyperparameter tune class weights

The heuristic for setting 'scale_pos_weight' parameter is effective for many situations.

Nevertheless, it is possible that better performance can be achieved with a different class weighting,
and this too will depend on the choice of performance metric used to evaluate the model.

For now, RandomizedSearchCV and GridSearchCV are used to search through a range of different class
weightings for class-weighted XGBoost and discover which results in the best ROC AUC score.
"""

"""
# RandomizedSearchCV tuning:
    
# Parameters to be used by RandomizedSearchCV using parameters from above-
rs_params = {
    'scale_pos_weight': sp_randint(1, 1000)
}

# Define model-
xgb_clf = xgb.XGBClassifier(
    objective = 'binary:logistic', eval_metric  = 'logloss',
    use_label_encoder = False)


import time
start_time = time.time()

# Initialize a RandomizedSearchCV object using 10-fold CV-
rs_cv = RandomizedSearchCV(
    estimator = xgb_clf, param_distributions=rs_params, n_iter = 100, cv = 3)

# Train RandomizedSearchCV object on training data-
rs_cv.fit(X_train, y_train)

end_time = time.time()

print(f"Best parameters found using RandomizedSearchCV: {rs_cv.best_params_}")
print(f"Time taken by RandomizedSearchCV = {end_time - start_time:.2f} seconds")
# Best parameters found using RandomizedSearchCV: {'scale_pos_weight': 4}
# Time taken by RandomizedSearchCV = 73.64 seconds


# Make predictions using RandomizedSearchCV-
y_pred_rs = rs_cv.predict(X_test)

# Get model metrics-
accuracy_rs = accuracy_score(y_test, y_pred_rs)
precision_rs = precision_score(y_test, y_pred_rs)
recall_rs = recall_score(y_test, y_pred_rs)
# f1score_rs = f1_score(y_test, y_pred_rs)

print("\nXGBoost classifier RandomizedSearchCV model metrics are:")
print(f"accuracy = {accuracy_rs * 100:.2f}%, precision = {precision_rs * 100:.2f}% & recall = {recall_rs * 100:.2f}%")
# XGBoost classifier RandomizedSearchCV model metrics are:
# Accuracy = 0.9970, Precision = 0.9565 & Recall = 0.7333

confusion_matrix(y_true = y_test, y_pred = y_pred_rs)
'''
array([[2969,    1],
       [   8,   22]], dtype=int64)
'''
"""



# GridSearchCV tuning:
    
# Parameters to be used by RandomizedSearchCV using parameters from above-
gs_params = {
    'scale_pos_weight': [1, 10, 25, 50, 75, 99, 100, 1000]
}

# Define model-
xgb_clf = xgb.XGBClassifier(
    objective = 'binary:logistic', eval_metric  = 'logloss',
    use_label_encoder = False)


import time
start_time = time.time()

# Initialize a RandomizedSearchCV object using 10-fold CV-
gs_cv = GridSearchCV(
    estimator = xgb_clf, param_grid = gs_params, cv = 3)

# Train GridSearchCV object on training data-
gs_cv.fit(X_train, y_train)

end_time = time.time()

print(f"Best parameters found using GridSearchCV: {gs_cv.best_params_}")
print(f"Time taken by GridSearchCV = {end_time - start_time:.2f} seconds")
# Best parameters found using GridSearchCV: {'scale_pos_weight': 1}
# Time taken by GridSearchCV = 5.00 seconds


# Make predictions using RandomizedSearchCV-
y_pred_gs = gs_cv.predict(X_test)

# Get model metrics-
accuracy_gs = accuracy_score(y_test, y_pred_gs)
precision_gs = precision_score(y_test, y_pred_gs)
recall_gs = recall_score(y_test, y_pred_gs)
# f1score_rs = f1_score(y_test, y_pred_rs)

print("\nXGBoost classifier GridSearchCV model metrics are:")
print(f"accuracy = {accuracy_gs * 100:.2f}%, precision = {precision_gs * 100:.2f}% & recall = {recall_gs * 100:.2f}%")
# XGBoost classifier GridSearchCV model metrics are:
# accuracy = 99.70%, precision = 95.65% & recall = 73.33%

confusion_matrix(y_true = y_test, y_pred = y_pred_gs)
'''
array([[2969,    1],
       [   8,   22]], dtype=int64)
'''

