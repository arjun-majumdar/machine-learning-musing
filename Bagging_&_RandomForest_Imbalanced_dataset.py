

"""
Bagging and Random Forest for Imbalanced Classification


Bagging is an ensemble algorithm that fits multiple models on different subsets of a training dataset and
then combines the predictions from all models.

Random forest is an extension of bagging that also randomly selects subsets of features used in each data
sample. Both bagging and random forests have proven effective on a wide range of different predictive
modeling problems.

Although effective, they are not suited to classification problems with a skewed class distribution.
Nevertheless, many modifications to the algorithms have been proposed that adapt their behavior and make
them better suited to a severe class imbalance.


Refer-
https://machinelearningmastery.com/bagging-and-random-forest-for-imbalanced-classification/
"""


"""
Bagging for Imbalanced Classification

Bootstrap Aggregation, or Bagging for short, is an ensemble ML algorithm.
It involves first selecting random samples of a training dataset with replacement, meaning that a given
sample may contain zero, one, or more than one copy of examples in the training dataset. This is called a
bootstrap sample. One weak learner model is then fit on each data sample. Typically, decision tree models
that do not use pruning (e.g. may overfit their training set slightly) are used as weak learners. Finally,
the predictions from all of the fit/trained weak learners are combined to make a single prediction (e.g.
aggregated).

'Each model in the ensemble is then used to generate a prediction for a new sample and these m predictions
are averaged to give the bagged model’s prediction.'
— Page 192, Applied Predictive Modeling, 2013.


The process of creating new bootstrap samples and fitting and adding trees to the sample can continue
until no further improvement is seen in the ensemble’s performance on a validation dataset.

This simple procedure often results in better performance than a single well-configured decision tree
algorithm.

Bagging as-is will create bootstrap samples that will not consider the skewed class distribution for
imbalanced classification datasets. As such, although the technique performs well in general, it may not
perform well if a severe class imbalance is present.
"""


from sklearn.ensemble import BaggingClassifier
import pandas as pd
import numpy as np
from numpy import mean
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
import matplotlib.pyplot as plt


"""
Standard Bagging

Before we dive into exploring extensions to bagging, let’s evaluate a standard bagged decision tree
ensemble without and use it as a point of comparison.

We can use the 'BaggingClassifier' scikit-sklearn class to create a bagged decision tree model with
roughly the same configuration.
"""
# Define a synthetic imbalanced binary classification problem with 10,000 examples, 99% of which are in
# the majority class and 1% are in the minority class-
X, y = make_classification(
    n_samples = 10000, n_features = 2,
    n_redundant = 0, n_clusters_per_class = 1,
    weights = [0.99], flip_y = 0
    )

X.shape, y.shape
# ((10000, 2), (10000,))

# Convert from np array to pd Series-
y_ser = pd.Series(y)

# Get target variable distribution-
y_ser.value_counts()
'''
0    9900
1     100
dtype: int64
'''

'''
uniq_val, count = np.unique(ar=y, return_counts=True)

uniq_val
# array([0, 1])

count
# array([9900,  100], dtype=int64)

# Create Python3 dict-
# WRONG - Redo
unique_values = {'unique_value': uniq_val, 'count': count}

unique_values
# {'unique_value': array([0, 1]), 'count': array([9900,  100], dtype=int64)}
'''


# Define the standard bagged decision tree ensemble model for evaluation-
model = BaggingClassifier()

# Evaluate model using repeated stratified k-fold CV with 3 repeats and 10 folds.
# Define evaluation procedure-
kfold_cv = RepeatedStratifiedKFold(n_splits = 10, n_repeats = 3)

# Evaluate model.
# We will use the mean ROC AUC score across all folds and repeats to evaluate the performance of the
# model.
scores = cross_val_score(model, X, y, scoring = 'roc_auc', cv = kfold_cv)

# Using accuracy as scoring metric-
scores_acc = cross_val_score(model, X, y, scoring = 'accuracy', cv = kfold_cv)

# Using precision and recall as scoring metrics-
scores_prec = cross_val_score(model, X, y, scoring = 'precision', cv = kfold_cv)
scores_rec = cross_val_score(model, X, y, scoring = 'recall', cv = kfold_cv)


scores.shape
# (30,)

print(f"Mean ROC-AUC = {np.mean(scores):.3f}")
# Mean ROC-AUC = 0.923

print(f"Mean accuracy = {np.mean(scores_acc) * 100:.3f}%")
# Mean accuracy = 99.157%

print(f"Mean precision = {np.mean(scores_prec) * 100:.3f}%")
print(f"Mean recall = {np.mean(scores_rec) * 100:.3f}%")
# Mean precision = 64.047%
# Mean recall = 51.000%


# For more scoring and metrics details, refer to-
# https://scikit-learn.org/stable/modules/model_evaluation.html




"""
Bagging With Random Undersampling

There are many ways to adapt bagging for use with imbalanced classification/dataset.

Perhaps the most straightforward approach is to apply data resampling on the bootstrap sample prior to
fitting/training the weak learner model. This might involve oversampling the minority class or
undersampling the majority class.

'An easy way to overcome class imbalance problem when facing the resampling stage in bagging is to take
the classes of the instances into account when they are randomly drawn from the original dataset.'
— Page 175, Learning from Imbalanced Data Sets, 2018.

Oversampling the minority class in the bootstrap is referred to as OverBagging; likewise, undersampling
the majority class in the bootstrap is referred to as UnderBagging, and combining both of the approaches
is referred to as OverUnderBagging.

The 'imbalanced-learn 'library provides an implementation of UnderBagging.

Specifically, it provides a version of bagging that uses a random undersampling strategy on the majority
class within a bootstrap sample in order to balance the two classes. This is provided in the
'BalancedBaggingClassifier' class.
"""
from imblearn.ensemble import BalancedBaggingClassifier

# Define model-
model_balanced_bagging_clf = BalancedBaggingClassifier(n_estimators = 100)
# The default number of trees (n_estimators) for this model and the previous is 10. In practice, it is
# a good idea to test larger values for this hyperparameter, such as 100 or 1,000.

# Although the 'BalancedBaggingClassifier' class uses a decision tree, you can test different models,
# such as k-nearest neighbors and more. You can set the 'base_estimator' argument when defining the
# class to use a different weak learner classifier model.


# We can evaluate a modified version of the bagged decision tree ensemble that performs random
# undersampling of the majority class prior to fitting each decision tree.
# We would expect that the use of random undersampling would improve the performance of the ensemble.
kfold_cv = RepeatedStratifiedKFold(n_splits = 10, n_repeats = 3)

scores = cross_val_score(model_balanced_bagging_clf, X, y, scoring = 'roc_auc', cv = kfold_cv)
scores_acc = cross_val_score(model_balanced_bagging_clf, X, y, scoring = 'accuracy', cv = kfold_cv)
scores_prec = cross_val_score(model_balanced_bagging_clf, X, y, scoring = 'precision', cv = kfold_cv)
scores_rec = cross_val_score(model_balanced_bagging_clf, X, y, scoring = 'recall', cv = kfold_cv)

print(f"Mean ROC-AUC using BalancedBaggingClassifier = {np.mean(scores):.3f}")
# Mean ROC-AUC using BalancedBaggingClassifier = 0.994

# In this example, we can see an increase on mean ROC AUC from about 0.923 without any data resampling,
# to about 0.994 with random undersampling of the majority class.

'''
This is not a true apples-to-apples comparison as we are using the same algorithm implementation from
two different libraries, but it makes the general point that balancing the bootstrap prior to fitting/
training a weak learner offers some benefit when the class distribution is skewed.
'''

print(f"Mean accuracy using BalancedBaggingClassifier = {np.mean(scores_acc) * 100:.3f}%")
print(f"Mean precision using BalancedBaggingClassifier = {np.mean(scores_prec) * 100:.3f}%")
print(f"Mean recall using BalancedBaggingClassifier = {np.mean(scores_rec) * 100:.3f}%")
# Mean accuracy using BalancedBaggingClassifier = 96.810%
# Mean precision using BalancedBaggingClassifier = 24.868%
# Mean recall using BalancedBaggingClassifier = 98.333%




"""
Random Forest for Imbalanced Classification

Random forest is another ensemble of decision tree models and may be considered an improvement upon
bagging.

Like bagging, random forest involves selecting bootstrap samples from the training dataset and fitting a
decision tree on each. The main difference is that all features (variables or columns) are not used;
instead, a small, randomly selected subset of features (columns) is chosen for each bootstrap sample.
This has the effect of de-correlating the decision trees (making them more independent), and in turn,
improving the ensemble prediction.

'Each model in the ensemble is then used to generate a prediction for a new sample and these m predictions
are averaged to give the forest’s prediction. Since the algorithm randomly selects predictors at each
split, tree correlation will necessarily be lessened.'
— Page 199, Applied Predictive Modeling, 2013.

Again, random forest is very effective on a wide range of problems, but like bagging, performance of the
standard algorithm is not great on imbalanced classification problems.

'In learning extremely imbalanced data, there is a significant probability that a bootstrap sample
contains few or even none of the minority class, resulting in a tree with poor performance for predicting
the minority class.'
— Using Random Forest to Learn Imbalanced Data, 2004.


Standard Random Forest

Before we dive into extensions of the random forest ensemble algorithm to make it better suited for
imbalanced classification, let’s fit and evaluate a random forest algorithm on our synthetic dataset.

We can use the RandomForestClassifier class from scikit-learn and use a small number of trees, in this
case, 100.
"""
from sklearn.ensemble import RandomForestClassifier

# Define model-
rf_clf = RandomForestClassifier(n_estimators = 100)

# Define evaluation procedure-
kfold_cv = RepeatedStratifiedKFold(n_splits = 10, n_repeats = 3)

# Evaluate model-
scores = cross_val_score(rf_clf, X, y, scoring = 'roc_auc', cv = kfold_cv)

scores = cross_val_score(rf_clf, X, y, scoring = 'roc_auc', cv = kfold_cv)
scores_acc = cross_val_score(rf_clf, X, y, scoring = 'accuracy', cv = kfold_cv)
scores_prec = cross_val_score(rf_clf, X, y, scoring = 'precision', cv = kfold_cv)
scores_rec = cross_val_score(rf_clf, X, y, scoring = 'recall', cv = kfold_cv)

print(f"Mean ROC-AUC using standard RF classifier = {np.mean(scores):.3f}")
print(f"Mean accuracy using standard RF classifier = {np.mean(scores_acc) * 100:.3f}%")
print(f"Mean precision using standard RF classifier = {np.mean(scores_prec) * 100:.3f}%")
print(f"Mean recall using standard RF classifier = {np.mean(scores_rec) * 100:.3f}%")
# Mean ROC-AUC using standard RF classifier = 0.973
# Mean accuracy using standard RF classifier = 99.247%
# Mean precision using standard RF classifier = 63.788%
# Mean recall using standard RF classifier = 50.333%




"""
Random Forest With Class Weighting

A simple technique for modifying a decision tree for imbalanced classification is to change the weight
that each class has when calculating the 'impurity' score of a chosen split point.

Impurity measures how mixed the groups of samples are for a given split in the training dataset and is
typically measured with Gini or entropy. The calculation can be biased so that a mixture in favor of the
minority class is favored, allowing some false positives for the majority class.

This modification of random forest is referred to as Weighted Random Forest.

'Another approach to make random forest more suitable for learning from extremely imbalanced data follows
the idea of cost sensitive learning. Since the RF classifier tends to be biased towards the majority
class, we shall place a heavier penalty on misclassifying the minority class.'
— Using Random Forest to Learn Imbalanced Data, 2004.

This can be achieved by setting the 'class_weight' argument on the 'RandomForestClassifier' class.
This argument takes a dictionary with a mapping of each class value (e.g. 0 and 1) to the weighting.
The argument value of 'balanced' can be provided to automatically use the inverse weighting from the
training dataset, giving focus to the minority class.
"""
# Define model-
rf_clf_balanced = RandomForestClassifier(n_estimators = 100, class_weight = 'balanced')

# Define evaluation procedure-
kfold_cv = RepeatedStratifiedKFold(n_splits = 10, n_repeats = 3)

# Evaluate model-
scores = cross_val_score(rf_clf_balanced, X, y, scoring = 'roc_auc', cv = kfold_cv)
scores_acc = cross_val_score(rf_clf_balanced, X, y, scoring = 'accuracy', cv = kfold_cv)
scores_prec = cross_val_score(rf_clf_balanced, X, y, scoring = 'precision', cv = kfold_cv)
scores_rec = cross_val_score(rf_clf_balanced, X, y, scoring = 'recall', cv = kfold_cv)

print(f"Mean ROC-AUC using balanced RF classifier = {np.mean(scores):.3f}")
print(f"Mean accuracy using balanced RF classifier = {np.mean(scores_acc) * 100:.3f}%")
print(f"Mean precision using balanced RF classifier = {np.mean(scores_prec) * 100:.3f}%")
print(f"Mean recall using balanced RF classifier = {np.mean(scores_rec) * 100:.3f}%")
# Mean ROC-AUC using balanced RF classifier = 0.983
# Mean accuracy using balanced RF classifier = 99.243%
# Mean precision using balanced RF classifier = 66.104%
# Mean recall using balanced RF classifier = 50.333%




"""
Random Forest With Bootstrap Class Weighting

Given that each decision tree is constructed from a bootstrap sample (e.g. random selection with
replacement), the class distribution in the data sample will be different for each tree.

As such, it might be interesting to change the class weighting based on the class distribution in each
bootstrap sample, instead of the entire training dataset.

This can be achieved by setting the 'class_weight' argument to the value 'balanced_subsample'.

We can test this modification and compare the results to the 'balanced' case above.
"""
# Define model-
rf_clf_balanced_sub = RandomForestClassifier(n_estimators = 100, class_weight = 'balanced_subsample')

# Define evaluation procedure-
kfold_cv = RepeatedStratifiedKFold(n_splits = 10, n_repeats = 3)

scores = cross_val_score(rf_clf_balanced_sub, X, y, scoring = 'roc_auc', cv = kfold_cv)
scores_acc = cross_val_score(rf_clf_balanced_sub, X, y, scoring = 'accuracy', cv = kfold_cv)
scores_prec = cross_val_score(rf_clf_balanced_sub, X, y, scoring = 'precision', cv = kfold_cv)
scores_rec = cross_val_score(rf_clf_balanced_sub, X, y, scoring = 'recall', cv = kfold_cv)

print(f"Mean ROC-AUC using balanced subsample RF classifier = {np.mean(scores):.3f}")
print(f"Mean accuracy using balanced subsample RF classifier = {np.mean(scores_acc) * 100:.3f}%")
print(f"Mean precision using balanced subsample RF classifier = {np.mean(scores_prec) * 100:.3f}%")
print(f"Mean recall using balanced subsample RF classifier = {np.mean(scores_rec) * 100:.3f}%")
# Mean ROC-AUC using balanced subsample RF classifier = 0.984
# Mean accuracy using balanced subsample RF classifier = 99.223%
# Mean precision using balanced subsample RF classifier = 66.666%
# Mean recall using balanced subsample RF classifier = 48.333%




"""
Random Forest With Random Undersampling

Another useful modification to random forest is to perform data resampling on the bootstrap sample in
order to explicitly change the class distribution.

The 'BalancedRandomForestClassifier' class from the 'imbalanced-learn' library implements this and
performs random undersampling of the majority class in reach bootstrap sample. This is generally referred
to as Balanced Random Forest.
"""
from imblearn.ensemble import BalancedRandomForestClassifier

# Define model-
bal_rf_clf = BalancedRandomForestClassifier(n_estimators = 100)
# We would expect this to have a more dramatic effect on model performance, given the broader success of
# data resampling techniques.

# Define evaluation procedure-
kfold_cv = RepeatedStratifiedKFold(n_splits = 10, n_repeats = 3)

scores = cross_val_score(bal_rf_clf, X, y, scoring = 'roc_auc', cv = kfold_cv)
scores_acc = cross_val_score(bal_rf_clf, X, y, scoring = 'accuracy', cv = kfold_cv)
scores_prec = cross_val_score(bal_rf_clf, X, y, scoring = 'precision', cv = kfold_cv)
scores_rec = cross_val_score(bal_rf_clf, X, y, scoring = 'recall', cv = kfold_cv)

print(f"Mean ROC-AUC using BalancedRandomForestClassifier = {np.mean(scores):.3f}")
print(f"Mean accuracy using BalancedRandomForestClassifier = {np.mean(scores_acc) * 100:.3f}%")
print(f"Mean precision using BalancedRandomForestClassifier = {np.mean(scores_prec) * 100:.3f}%")
print(f"Mean recall using BalancedRandomForestClassifier = {np.mean(scores_rec) * 100:.3f}%")
# Mean ROC-AUC using BalancedRandomForestClassifier = 0.994
# Mean accuracy using BalancedRandomForestClassifier = 96.087%
# Mean precision using BalancedRandomForestClassifier = 20.486%
# Mean recall using BalancedRandomForestClassifier = 99.333%




"""
Easy Ensemble for Imbalanced Classification

When considering bagged ensembles for imbalanced classification, a natural thought might be to use random
resampling of the majority class to create multiple datasets with a balanced class distribution.

Specifically, a dataset can be created from all of the examples in the minority class and a randomly
selected sample from the majority class. Then a model or weak learner can be fit on this dataset. The
process can be repeated multiple times and the average prediction across the ensemble of models can be
used to make predictions.

This is exactly the approach proposed by Xu-Ying Liu, et al. in their 2008 paper titled 'Exploratory
Undersampling for Class-Imbalance Learning.'

The selective construction of the subsamples is seen as a type of undersampling of the majority class.
The generation of multiple subsamples allows the ensemble to overcome the downside of undersampling in
which valuable information is discarded from the training process.

'… under-sampling is an efficient strategy to deal with class-imbalance. However, the drawback of
under-sampling is that it throws away many potentially useful data.'
— Exploratory Undersampling for Class-Imbalance Learning, 2008.

The authors propose variations on the approach, such as the Easy Ensemble and the Balance Cascade.

Let’s take a closer look at the Easy Ensemble.




Easy Ensemble

The Easy Ensemble involves creating balanced samples of the training dataset by selecting all examples
from the minority class and a subset from the majority class.

Rather than using pruned decision trees, boosted decision trees are used on each subset, specifically the
AdaBoost algorithm.

AdaBoost works by first fitting a decision tree on the dataset, then determining the errors made by the
tree and weighing the examples in the dataset by those errors so that more attention is paid to the
misclassified examples and less to the correctly classified examples. A subsequent tree is then fit on the
weighted dataset intended to correct the errors. The process is then repeated for a given number of
decision trees.

'This means that samples that are difficult to classify receive increasingly larger weights until the
algorithm identifies a model that correctly classifies these samples. Therefore, each iteration of the
algorithm is required to learn a different aspect of the data, focusing on regions that contain
difficult-to-classify samples.'
— Page 389, Applied Predictive Modeling, 2013.

The 'EasyEnsembleClassifier' class from the imbalanced-learn library provides an implementation of the
easy ensemble technique.
"""
from imblearn.ensemble import EasyEnsembleClassifier

# Define model-
easy_clf = EasyEnsembleClassifier(n_estimators = 100)
# Given the use of a type of random undersampling, we would expect the technique to perform well in
# general.
# Although an AdaBoost classifier is used on each subsample, alternate classifier models can be used
# via setting the 'base_estimator' argument in the model.

# Define evaluation procedure-
kfold_cv = RepeatedStratifiedKFold(n_splits = 10, n_repeats = 3)

scores = cross_val_score(easy_clf, X, y, scoring = 'roc_auc', cv = kfold_cv)
scores_acc = cross_val_score(easy_clf, X, y, scoring = 'accuracy', cv = kfold_cv)
scores_prec = cross_val_score(easy_clf, X, y, scoring = 'precision', cv = kfold_cv)
scores_rec = cross_val_score(easy_clf, X, y, scoring = 'recall', cv = kfold_cv)

print(f"Mean ROC-AUC using EasyEnsembleClassifier = {np.mean(scores):.3f}")
print(f"Mean accuracy using EasyEnsembleClassifier = {np.mean(scores_acc) * 100:.3f}%")
print(f"Mean precision using EasyEnsembleClassifier = {np.mean(scores_prec) * 100:.3f}%")
print(f"Mean recall using EasyEnsembleClassifier = {np.mean(scores_rec) * 100:.3f}%")
# Mean ROC-AUC using EasyEnsembleClassifier = 0.994
# Mean accuracy using EasyEnsembleClassifier = 96.933%
# Mean precision using EasyEnsembleClassifier = 25.192%
# Mean recall using EasyEnsembleClassifier = 100.000%



