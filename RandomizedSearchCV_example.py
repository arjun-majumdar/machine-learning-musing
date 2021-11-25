

"""
Hyperparameter Optimization With Random Search and Grid Search

ML models have hyperparameters that you must set in order to customize the model to your dataset.
Often the general effects of hyperparameters on a model are known, but how to best set a
hyperparameter and combinations of interacting hyperparameters for a given dataset is challenging.
There are often general heuristics or rules of thumb for configuring hyperparameters.

A better approach is to objectively search different values for model hyperparameters and choose a
subset that results in a model that achieves the best performance on a given dataset. This is called
hyperparameter optimization or hyperparameter tuning and is available in the scikit-learn Python ML
library. The result of a hyperparameter optimization is a single set of well-performing hyperparameters
that you can use to configure your model.


To find the set of hyperparameters giving the best result RandomizedSearchCV can be used.
RandomizedSearchCV randomly passes the set of hyperparameters and calculates the score and gives
the best set of hyperparameters which gives the best score as an output. 

Refer-
https://machinelearningmastery.com/hyperparameter-optimization-with-random-search-and-grid-search/  
https://www.projectpro.io/recipes/find-optimal-parameters-using-randomizedsearchcv-for-regression
"""


'''
Model Hyperparameter Optimization

ML models have hyperparameters. Hyperparameters are points of choice or configuration that allow a
ML model to be customized for a specific task or dataset.

Hyperparameter: Model configuration argument specified by the developer to guide the learning process
for a specific dataset.

ML models also have parameters, which are the internal coefficients set by training or optimizing the
model on a training dataset.
Parameters are different from hyperparameters. Parameters are learned automatically; hyperparameters
are set manually to help guide the learning process. For more on the difference between parameters
and hyperparameters, see the tutorial-
https://machinelearningmastery.com/difference-between-a-parameter-and-a-hyperparameter/

Typically a hyperparameter has a known effect on a model in the general sense, but it is not clear
how to best set a hyperparameter for a given dataset. Further, many ML models have a range of
hyperparameters and they may interact in nonlinear ways.

As such, it is often required to search for a set of hyperparameters that result in the best performance
of a model for a given dataset. This is called hyperparameter optimization, hyperparameter tuning, or
hyperparameter search.

An optimization procedure involves defining a search space. This can be thought of geometrically as an
n-dimensional volume, where each hyperparameter represents a different dimension and the scale of the
dimension are the values that the hyperparameter may take on, such as real-valued, integer-valued, or
categorical.

Search Space: Volume to be searched where each dimension represents a hyperparameter and each point
represents one model configuration.

A point in the search space is a vector with a specific value for each hyperparameter value. The goal
of the optimization procedure is to find a vector that results in the best performance of the model after
learning, such as maximum accuracy or minimum error.

A range of different optimization algorithms may be used, although two of the simplest and most common
methods are random search and grid search.

1. Random Search. Define a search space as a bounded domain of hyperparameter values and randomly sample
points in that domain.

2. Grid Search. Define a search space as a grid of hyperparameter values and evaluate every position in
the grid.

Grid search is great for spot-checking combinations that are known to perform well generally. Random
search is great for discovery and getting hyperparameter combinations that you would not have guessed
intuitively, although it often requires more time to execute.

More advanced methods are sometimes used, such as Bayesian Optimization and Evolutionary Optimization.
https://machinelearningmastery.com/what-is-bayesian-optimization/


Both classes require two arguments. The first is the model that you are optimizing. This is an instance
of the model with values of hyperparameters set that you want to optimize. The second is the search space.
This is defined as a dictionary where the names are the hyperparameter arguments to the model and the
values are discrete values or a distribution of values to sample in the case of a random search.

# define model
# model = LogisticRegression()

# define search space
# space = dict()

# define search
# search = GridSearchCV(model, space)

Both classes provide a 'cv' argument that allows either an integer number of folds to be specified,
e.g. 5, or a configured cross-validation object. Author recommends defining and specifying a
cross-validation object to gain more control over model evaluation and make the evaluation procedure
obvious and explicit.

In the case of classification tasks, I recommend using the 'RepeatedStratifiedKFold' class, and for
regression tasks, author recommends using the 'RepeatedKFold' with an appropriate number of folds and
repeats, such as 10 folds and three repeats.

# define evaluation
# cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# define search
# search = GridSearchCV(..., cv=cv)

For more details refer to-
https://stackoverflow.com/questions/65318931/stratifiedkfold-vs-kfold-in-scikit-learn
https://stackoverflow.com/questions/65318931/stratifiedkfold-vs-kfold-in-scikit-learn
https://xzz201920.medium.com/stratifiedkfold-v-s-kfold-v-s-stratifiedshufflesplit-ffcae5bfdf


Both hyperparameter optimization classes also provide a 'scoring' argument that takes a string indicating
the metric to optimize. The metric must be maximizing, meaning better models result in larger scores.
For classification, this may be ‘accuracy‘. For regression, this is a negative error measure, such as
‘neg_mean_absolute_error‘ for a negative version of the mean absolute error, where values closer to zero
represent less prediction error by the model.

# define search
# search = GridSearchCV(..., scoring='neg_mean_absolute_error')

You can see a list of build-in scoring metrics here-
https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter

Finally, the search can be made parallel, e.g. use all of the CPU cores by specifying the 'n_jobs'
argument as an integer with the number of cores in your system, e.g. 8. Or you can set it to be -1 to
automatically use all of the cores in your system.

# define search
# search = GridSearchCV(..., n_jobs=-1)

Once defined, the search is performed by calling the 'fit()' function and providing a dataset used to
train and evaluate model hyperparameter combinations using cross-validation.

# execute search
# result = search.fit(X, y)

Running the search may take minutes or hours, depending on the size of the search space and the speed of
your hardware. You will often want to tailor the search to how much time you have rather than the
possibility of what could be searched.

At the end of the search, you can access all of the results via the attributes of the class. Perhaps the
most important attributes are the best score observed and the hyperparameters that achieved the best score.

# summarize result
# print('Best Score: %s' % result.best_score_)
# print('Best Hyperparameters: %s' % result.best_params_)

Once you know the set of hyperparameters that achieve the best result, you can then define a new model,
set the values of each hyperparameter, then fit the model on all available data. This model can then be
used to make predictions on new data.
'''


from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from scipy.stats import uniform as sp_randFloat
from scipy.stats import randint as sp_randInt
from scipy.stats import loguniform
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Load sklearn dataset-
dataset = datasets.load_diabetes()

# Extract features (X) and target (y)-
X = dataset.data
y = dataset.target

# Convert from np array to pd DataFrame-
X_df = pd.DataFrame(X, columns = dataset.feature_names)

X_df.shape, y.shape
# ((442, 10), (442,))

X_df.head()
'''
        age       sex       bmi        bp        s1        s2        s3        s4        s5        s6
0  0.038076  0.050680  0.061696  0.021872 -0.044223 -0.034821 -0.043401 -0.002592  0.019908 -0.017646
1 -0.001882 -0.044642 -0.051474 -0.026328 -0.008449 -0.019163  0.074412 -0.039493 -0.068330 -0.092204
2  0.085299  0.050680  0.044451 -0.005671 -0.045599 -0.034194 -0.032356 -0.002592  0.002864 -0.025930
3 -0.089063 -0.044642 -0.011595 -0.036656  0.012191  0.024991 -0.036038  0.034309  0.022692 -0.009362
4  0.005383 -0.044642 -0.036385  0.021872  0.003935  0.015596  0.008142 -0.002592 -0.031991 -0.046641
'''
# NOTE: data seems to be already scaled.

del X

# Split features & target into train & test sets-
X_train, X_test, y_train, y_test = train_test_split(
    X_df, y,
    test_size = 0.3
    )

X_train.shape, y_train.shape
# ((309, 10), (309,))

X_test.shape, y_test.shape
# ((133, 10), (133,))


# Visualize distribution of target attribute-
num, bins, patches = plt.hist(y, bins = int(np.ceil(np.sqrt(y.size))))
plt.show()

# Visualize distributions of all numeric attributes in features-
sns.boxplot(data = X_df)
plt.title("Pima diabetes: Boxplot distribution - numeric columns")
plt.show()


# Initialize a GradientBoostingRegressor model-
gbr = GradientBoostingRegressor()

# Specify parameters for hyper-parameter tuning-
parameters = {
    'learning_rate': sp_randFloat(),
    'subsample'    : sp_randFloat(),
    'n_estimators' : sp_randInt(100, 1000),
    'max_depth'    : sp_randInt(4, 10)
    }


'''
RandomizedSearchCV parameters-

1. estimator: here, we input the metric or the model for which we need to optimize the parameters.

2. param_distributions: here, we have to pass the dictionary of parameters that we need to optimize.

3. cv: here, we have to pass an interger value signifying the number of splits needed for cross
validation. By default it's 5.

3. n_iter: Number of parameter settings that are sampled. n_iter trades off runtime vs quality
of the solution

4. scoring: str, callable, list, tuple or dict, default = None.
Strategy to evaluate the performance of the cross-validated model on the test set. 
Refer-
https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter

5. return_train_score: bool, default=False
If False, the 'cv_results_' attribute will not include training scores. Computing training scores
is used to get insights on how different parameter settings impact the overfitting/underfitting
trade-off. However computing the scores on the training set can be computationally expensive and
is not strictly required to select the parameters that yield the best generalization performance.

6. n_jobs: This signifies the number of jobs to be run in parallel, -1 signifies to use all processor.
'''

# Refer-
# https://stackoverflow.com/questions/65318931/stratifiedkfold-vs-kfold-in-scikit-learn
# https://stackoverflow.com/questions/65318931/stratifiedkfold-vs-kfold-in-scikit-learn
# https://xzz201920.medium.com/stratifiedkfold-v-s-kfold-v-s-stratifiedshufflesplit-ffcae5bfdf

# Define a cross-validation object for regression task-
cv_obj = RepeatedKFold(n_splits = 3, n_repeats = 2)

# Initialize a RandomizedSearchCV object-
rnd_srch_cv = RandomizedSearchCV(
    estimator = gbr, param_distributions = parameters,
    # cv = 3,
    cv = cv_obj,
    n_iter = 500,
    return_train_score = True,
    scoring = 'r2'
    )

# Perform RandomizedSearchCV on training data-
rnd_srch_cv.fit(X_train, y_train)

print(f"'best' hyper-parameters found:\n{rnd_srch_cv.best_params_}")
"""
'best' hyper-parameters found:
{'learning_rate': 0.03161782403928204, 'max_depth': 8, 'n_estimators': 164, 'subsample': 0.2496627490045018}
{'learning_rate': 0.01972570433834031, 'max_depth': 4, 'n_estimators': 118, 'subsample': 0.4946858413460623}
"""

print(f"'best' R2-Score = {rnd_srch_cv.best_score_:.4f}")
# 'best' R2-Score = 0.4516
# 'best' R2-Score = 0.4570




# Hyper-parameter tuning for classification:

# Load dataset for classification-
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/sonar.csv'
data = pd.read_csv(url, header = None)

data.shape
# (208, 61)

# Split data into features (X) and labels (y)-
X, y = data.iloc[:, :-1], data.iloc[:, -1]

X.shape, y.shape
# ((208, 60), (208,))

# Target attribte distribution-
y.value_counts()
'''
M    111
R     97
Name: 60, dtype: int64
'''

le = LabelEncoder()
y = le.fit_transform(y)

# Split features & target into train & test sets-
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size = 0.3, stratify = y
    )

X_train.shape, y_train.shape
# ((145, 60), (145,))

X_test.shape, y_test.shape
# ((63, 60), (63,))


# Use a LogisticRegression classifier-
log_reg_clf = LogisticRegression()
log_reg_clf.fit(X_train, y_train)

y_pred = log_reg_clf.predict(X_test)
acc = accuracy_score(y_true = y_test, y_pred = y_pred)
prec = precision_score(y_true = y_test, y_pred = y_pred)
rec = recall_score(y_true = y_test, y_pred = y_pred)

print("LogisticRegression (base) classifier validation metrics:")
print(f"accuracy = {acc * 100:.2f}%, precision = {prec * 100:.2f}% & recall = {rec * 100:.2f}%")
# LogisticRegression (base) classifier validation metrics:
# accuracy = 77.78%, precision = 75.86% & recall = 75.86%

print(f"LogisticRegression (base) classifier confusion matrix:\n"
f"{confusion_matrix(y_true = y_test, y_pred = y_pred)}")
'''
LogisticRegression (base) classifier confusion matrix:
[[27  7]
 [ 7 22]]
'''


# Hyper-parameter tune with RandomizedSearchCV:

# Use repeated stratified k-fold cross-validation with 3 repeats and 10 folds-
cv_obj = RepeatedStratifiedKFold(n_splits = 10, n_repeats = 3)

# Define hyper-parameter search space-
space = {}
space['solver'] = ['newton-cg', 'lbfgs', 'liblinear']
space['penalty'] = ['none', 'l1', 'l2', 'elasticnet']
space['C'] = loguniform(1e-5, 100)

# Initialize a RandomizedSearchCV object and train on training data-
rand_srch_cv = RandomizedSearchCV(
    estimator = LogisticRegression(), param_distributions = space,
    n_iter = 500, cv = cv_obj,
    scoring = 'accuracy'
    )

output = rand_srch_cv.fit(X_train, y_train)

print(f"'Best' parameters found with RandomizedSearchCV are:\n{output.best_params_}")
'''
'Best' parameters found with RandomizedSearchCV are:
{'C': 6.3471968762365645, 'penalty': 'l1', 'solver': 'liblinear'}
'''

print(f"'Best' accuracy score achieved with RandomizedSearchCV = {output.best_score_ * 100:.2f}%")
# 'Best' accuracy score achieved with RandomizedSearchCV = 81.78%

# Sanity check - initialize and train a 'best' model using parameters achieved from above-
best_log_reg_clf = LogisticRegression(
    C = 6.3471968762365645, penalty = 'l1',
    solver = 'liblinear')
best_log_reg_clf.fit(X_train, y_train)

y_pred_best = best_log_reg_clf.predict(X_test)
acc = accuracy_score(y_true = y_test, y_pred = y_pred_best)
prec = precision_score(y_true = y_test, y_pred = y_pred_best)
rec = recall_score(y_true = y_test, y_pred = y_pred_best)

print("LogisticRegression (best - RandomizedSearchCV) classifier validation metrics:")
print(f"accuracy = {acc * 100:.2f}%, precision = {prec * 100:.2f}% & recall = {rec * 100:.2f}%")
# LogisticRegression (best - RandomizedSearchCV) classifier validation metrics:
# accuracy = 76.19%, precision = 71.88% & recall = 79.31%

print(f"LogisticRegression (best - RandomizedSearchCV) classifier confusion matrix:\n"
f"{confusion_matrix(y_true = y_test, y_pred = y_pred_best)}")
'''
LogisticRegression (best - RandomizedSearchCV) classifier confusion matrix:
[[25  9]
 [ 6 23]]
'''




# Use RandomForestClassifier-
rfc = RandomForestClassifier(n_estimators = 50)
rfc.fit(X_train, y_train)

y_pred = rfc.predict(X_test)
acc = accuracy_score(y_true = y_test, y_pred = y_pred)
prec = precision_score(y_true = y_test, y_pred = y_pred)
rec = recall_score(y_true = y_test, y_pred = y_pred)

print("RandomForestClassifier (base; n_estimators = 50) validation metrics:")
print(f"accuracy = {acc * 100:.2f}%, precision = {prec * 100:.2f}% & recall = {rec * 100:.2f}%")
# RandomForestClassifier (base; n_estimators = 50) validation metrics:
# accuracy = 82.54%, precision = 78.12% & recall = 86.21%

print(f"confusion matrix:\n{confusion_matrix(y_true = y_test, y_pred = y_pred)}")
'''
confusion matrix:
[[27  7]
 [ 4 25]]
'''

