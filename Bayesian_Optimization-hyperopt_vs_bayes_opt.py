

"""
Bayesian Optimization: bayes_opt or hyperopt

Bayesian Optimization also runs ML models many times with different sets of hyperparameter
values, but it evaluates the past model information to select hyperparameter values to
build/train the newer ML model. This is said to spend less time to reach the highest accuracy
model than other methods.


There are two packages in Python that author uses for Bayesian Optimization- 'bayes_opt' and
'hyperopt'.


Refer-
https://www.analyticsvidhya.com/blog/2021/05/bayesian-optimization-bayes_opt-or-hyperopt/
"""


'''
Problem With Uninformed Search

The problem with uninformed search is that it takes a relatively long time to build all the models.
Informed search can solve this problem. In informed search, the previous models with a certain set
of hyperparameter values can inform the later model which hyperparameter values are better to select.

One of the methods to do this is coarse-to-fine. This involves running GridSearchCV or
RandomizedSearchCV more than once. Each time, the hyperparameter value range is more specific.

For example, we start RandomizedSearchCV say with 'learning_rate' parameter ranging from 0.01 to 1.
Then, we find out that high accuracy models have their 'learning_rate' around 0.1 to 0.3. Hence, we
can run again 'GridSearchCV' focusing on the 'learning_rate' between 0.1 and 0.3. This process can
continue until a satisfactory result is achieved. The first trial is coarse because the value range
is large, from '0.01 to 1'. The later trial is fine as the value range is focused on '0.1' to '0.3'.

The drawback of the coarse-to-fine method is that we need to run the code repeatedly and observe the
value range of hyper parameters being tuned. You might be thinking if there is a way to automate this
and one of the way is Bayesian Optimization.
'''


from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
# from scipy.stats import uniform as sp_randFloat
# from scipy.stats import randint as sp_randInt
# from scipy.stats import loguniform
from bayes_opt import BayesianOptimization
from hyperopt import hp, fmin, tpe
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time


# Load sklearn dataset-
dataset = datasets.load_wine()

# Extract features (X) and target (y)-
X = dataset.data
y = dataset.target

# Convert from np array to pd DataFrame-
X_df = pd.DataFrame(X, columns = dataset.feature_names)

X_df.shape, y.shape
# ((178, 13), (178,))

X_df.head()
'''
   alcohol  malic_acid   ash  alcalinity_of_ash  ...  color_intensity   hue  od280/od315_of_diluted_wines  proline
0    14.23        1.71  2.43               15.6  ...             5.64  1.04                          3.92   1065.0
1    13.20        1.78  2.14               11.2  ...             4.38  1.05                          3.40   1050.0
2    13.16        2.36  2.67               18.6  ...             5.68  1.03                          3.17   1185.0
3    14.37        1.95  2.50               16.8  ...             7.80  0.86                          3.45   1480.0
4    13.24        2.59  2.87               21.0  ...             4.32  1.04                          2.93    735.0

[5 rows x 13 columns]
'''
# NOTE: data seems to be already scaled.

del X

X_df.columns
'''
Index(['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium',
       'total_phenols', 'flavanoids', 'nonflavanoid_phenols',
       'proanthocyanins', 'color_intensity', 'hue',
       'od280/od315_of_diluted_wines', 'proline'],
      dtype='object')
'''

# Get distribution of target attribute-
np.unique(y, return_counts = True)
# (array([0, 1, 2]), array([59, 71, 48], dtype=int64))

# Visualize distribution of target attribute-
num, bins, patches = plt.hist(y, bins = int(np.ceil(np.sqrt(y.size))))
plt.show()

# Visualize distributions of all numeric attributes in features-
sns.boxplot(data = X_df)
plt.title("Wine classification: Boxplot distribution - numeric columns")
plt.xticks(rotation = 60)
plt.show()

X_df.describe()
'''
          alcohol  malic_acid         ash  alcalinity_of_ash   magnesium  total_phenols  flavanoids  nonflavanoid_phenols  proanthocyanins  color_intensity         hue  od280/od315_of_diluted_wines      proline
count  178.000000  178.000000  178.000000         178.000000  178.000000     178.000000  178.000000            178.000000       178.000000       178.000000  178.000000                    178.000000   178.000000
mean    13.000618    2.336348    2.366517          19.494944   99.741573       2.295112    2.029270              0.361854         1.590899         5.058090    0.957449                      2.611685   746.893258
std      0.811827    1.117146    0.274344           3.339564   14.282484       0.625851    0.998859              0.124453         0.572359         2.318286    0.228572                      0.709990   314.907474
min     11.030000    0.740000    1.360000          10.600000   70.000000       0.980000    0.340000              0.130000         0.410000         1.280000    0.480000                      1.270000   278.000000
25%     12.362500    1.602500    2.210000          17.200000   88.000000       1.742500    1.205000              0.270000         1.250000         3.220000    0.782500                      1.937500   500.500000
50%     13.050000    1.865000    2.360000          19.500000   98.000000       2.355000    2.135000              0.340000         1.555000         4.690000    0.965000                      2.780000   673.500000
75%     13.677500    3.082500    2.557500          21.500000  107.000000       2.800000    2.875000              0.437500         1.950000         6.200000    1.120000                      3.170000   985.000000
max     14.830000    5.800000    3.230000          30.000000  162.000000       3.880000    5.080000              0.660000         3.580000        13.000000    1.710000                      4.000000  1680.000000
'''

# Scale numeric columns with a StandardScaler-
std_scaler = StandardScaler()
X_df_scaled = std_scaler.fit_transform(X_df)

X_df.shape, X_df_scaled.shape
# ((178, 13), (178, 13))


# Split features & target into train & test sets-
X_train, X_test, y_train, y_test = train_test_split(
    X_df_scaled, y,
    test_size = 0.3, stratify = y
    )

X_train.shape, y_train.shape
# ((124, 13), (124,))

X_test.shape, y_test.shape
# ((54, 13), (54,))




# It is recommended to definine and specify a cross-validation object to gain more
# control over model evaluation and make the evaluation procedure obvious and explicit.
# For classification tasks - 'RepeatedStratifiedKFold'
# For regression tasks - 'RepeatedKFold'

# Initialize a cross-validation object-
rptd_skf = RepeatedStratifiedKFold(n_splits = 3, n_repeats = 2)




# 'bayes_opt' package for Bayesian Optimization
# Refer-
# https://github.com/fmfn/BayesianOptimization

# Specify parameters for hyper-parameter tuning with their corresponding boungs-
# Bounded region of parameter space
parameters = {
    'learning_rate': (0.01, 0.95),
    'subsample'    : (0.5, 1),
    'n_estimators' : (50, 100),
    'max_depth'    : (3, 11),
    'max_features': (0.6, 1)
    }


def grad_boost_machine_clf_optimzation_fn(n_estimators, learning_rate, max_features, max_depth, subsample):
    # specify & define a function to be optimized
    
    # Python3 dict to contain parameters-
    params = {}
    params['n_estimators'] = round(n_estimators)
    params['learning_rate'] = learning_rate
    params['max_features'] = max_features
    params['max_depth'] = round(max_depth)
    params['subsample'] = subsample
    
    scores = cross_val_score(
        GradientBoostingClassifier(**params),
        X_train, y_train, scoring = 'accuracy',
        cv = rptd_skf).mean()
    return scores.mean()


# Instantiate a 'BayesianOptimization' object specifying a function to be optimized 'f',
# and its parameters with their corresponding bounds, 'pbounds'. This is a constrained
# optimization technique, so you must specify the minimum and maximum values that can be
# probed for each parameter in order for it to work-
bayesian_opt_gbm_clf = BayesianOptimization(
    f = grad_boost_machine_clf_optimzation_fn,
    pbounds = parameters
    )

'''
The 'BayesianOptimization' object will work out of the box without much tuning needed.
The main method you should be aware of is 'maximize()', which does exactly what you think it does.

There are many parameters you can pass to 'maximize()', nonetheless, the most important ones are-

1. n_iter: How many steps of bayesian optimization you want to perform. The more steps the more likely
you are to find a good maximum.

2. init_points: How many steps of random exploration you want to perform. Random exploration can help
by diversifying the exploration space.
'''

start_time = time.time()

bayesian_opt_gbm_clf.maximize(
    init_points = 30,
    n_iter = 10
)

end_time = time.time()
print(f"Time taken for Bayesian Optimization using 'bayes_opt' = {end_time - start_time:4f} seconds")
# Time taken for Bayesian Optimization using 'bayes_opt' = 33.7960 seconds

print(f"Best set of hyper-parameters found:\n{bayesian_opt_gbm_clf.max['params']}")
'''
Best set of hyper-parameters found:
{'learning_rate': 0.03198860324531767, 'max_depth': 5.742352637146139, 'max_features': 0.6136802519870926, 'n_estimators': 95.04103955629935, 'subsample': 0.6126515646224074}
'''

print(f"Best accuracy achieved = {bayesian_opt_gbm_clf.max['target'] * 100:.2f}%")
# Best accuracy achieved = 98.39%


# Sanity check- train a new model with the hyper-parameter tuned parameters-
gbm_clf = GradientBoostingClassifier(
    learning_rate = bayesian_opt_gbm_clf.max['params']['learning_rate'],
    max_depth = round(bayesian_opt_gbm_clf.max['params']['max_depth']),
    max_features = bayesian_opt_gbm_clf.max['params']['max_features'],
    n_estimators = round(bayesian_opt_gbm_clf.max['params']['n_estimators']),
    subsample = bayesian_opt_gbm_clf.max['params']['subsample']
    )

gbm_clf.fit(X_train, y_train)
y_pred = gbm_clf.predict(X_test)

acc = accuracy_score(y_true = y_test, y_pred = y_pred)
prec = precision_score(y_true = y_test, y_pred = y_pred, average = 'macro')
rec = recall_score(y_true = y_test, y_pred = y_pred, average = 'macro')

print("Hyper-parameter tuned GradientBoostingClassifier metrics:")
print(f"accuracy = {acc * 100:.2f}%, precision = {prec * 100:.2f}% & recall = {rec * 100:.2f}%")
# Hyper-parameter tuned GradientBoostingClassifier metrics:
# accuracy = 100.00%, precision = 100.00% & recall = 100.00%

print(f"Hyper-parameter tuned GradientBoostingClassifier confusion matrix:\n{confusion_matrix(y_true = y_test, y_pred = y_pred)}")
'''
Hyper-parameter tuned GradientBoostingClassifier confusion matrix:
[[18  0  0]
 [ 0 21  0]
 [ 0  0 15]]
'''

# NOTE: Strange to get 100% accuracy, precision and recall scores!




# Another Python package to leverage Bayesian Optimization is 'hyperopt'
# Refer to-
# http://hyperopt.github.io/hyperopt/
# "Algorithms for Hyper-Parameter Optimization" by James Bergstra et al. research paper

# https://towardsdatascience.com/hyperopt-hyperparameter-tuning-based-on-bayesian-optimization-7fa32dffaf29

# Specify hyper-parameters for 'hyperopt' package-
parameters_hyperopt = {
    'learning_rate': hp.uniform('learning_rate', 0.01, 0.95),
    'subsample': hp.uniform('subsample', 0.5, 1),
    'n_estimators': hp.randint('n_estimators', 50, 100),
    'max_depth': hp.randint('max_depth', 3, 11),
    'max_features': hp.uniform('max_features', 0.6, 1)

}


def gbm_clf_optim_fn(params):
    # Define an objective function to be optimized.
    
    # Python3 dict to contain parameters-
    params = {
        'n_estimators': params['n_estimators'],
        'learning_rate': params['learning_rate'],
        'max_features': params['max_features'],
        'max_depth': params['max_depth'],
        'subsample': params['subsample']
    }
    
    scores = cross_val_score(
        GradientBoostingClassifier(**params),
        X_train, y_train, scoring = 'accuracy',
        cv = rptd_skf).mean()
    # return scores.mean()
    return 1 - scores


start_time = time.time()

gbm_clf_best_params_hyperopt = fmin(
    fn = gbm_clf_optim_fn, space = parameters_hyperopt,
    max_evals = 30, algo = tpe.suggest
    )

end_time = time.time()
# 100%|██████████████████████████████████████████████| 30/30 [00:16<00:00,  1.78trial/s, best loss: 0.012001548586914379]

print(f"Time taken for Bayesian Optimization using 'hyperopt' = {end_time - start_time:4f} seconds")
# Time taken for Bayesian Optimization using 'hyperopt' = 16.874259 seconds

print(f"Best hyper-parameters found with 'hyperopt' are:\n{gbm_clf_best_params_hyperopt}")
'''
Best hyper-parameters found with 'hyperopt' are:
{'learning_rate': 0.4675068748581098, 'max_depth': 8, 'max_features': 0.8332466185795476, 'n_estimators': 92, 'subsample': 0.5801410026577661}
'''

# The best loss is 0.012 implying that the 'best' achieved accuracy = 1 - 0.012 = 0.988 = 98.8%.


# Sanity check- train a new model with the hyper-parameter tuned parameters-
best_gbm_clf = GradientBoostingClassifier(
    learning_rate = gbm_clf_best_params_hyperopt['learning_rate'],
    max_depth = gbm_clf_best_params_hyperopt['max_depth'],
    max_features = gbm_clf_best_params_hyperopt['max_features'],
    n_estimators = gbm_clf_best_params_hyperopt['n_estimators'],
    subsample = gbm_clf_best_params_hyperopt['subsample']
    )

best_gbm_clf.fit(X_train, y_train)
y_pred = best_gbm_clf.predict(X_test)

acc = accuracy_score(y_true = y_test, y_pred = y_pred)
prec = precision_score(y_true = y_test, y_pred = y_pred, average = 'macro')
rec = recall_score(y_true = y_test, y_pred = y_pred, average = 'macro')

print("Hyper-parameter tuned GradientBoostingClassifier (hyperopt) metrics:")
print(f"accuracy = {acc * 100:.2f}%, precision = {prec * 100:.2f}% & recall = {rec * 100:.2f}%")
# Hyper-parameter tuned GradientBoostingClassifier (hyperopt) metrics:
# accuracy = 98.15%, precision = 98.25% & recall = 98.41%

print(f"Hyper-parameter tuned GradientBoostingClassifier (hyperopt) confusion matrix:\n{confusion_matrix(y_true = y_test, y_pred = y_pred)}")
'''
Hyper-parameter tuned GradientBoostingClassifier (hyperopt) confusion matrix:
[[18  0  0]
 [ 1 20  0]
 [ 0  0 15]]
'''




'''
'bayes_opt' package shows the process of tuning the values of the hyperparameters. We can visualize
which values are used for each iteration. 'hyperopt' package only shows one line of the progress bar
which is the best loss, and the duration.

'bayes_opt' can be preferred because, in reality, the hyper-parameter tuning process might take a long
time wherein we may want to terminate the process. After doing so, we might want to use the best
hyperparameter-tuning result obtained so far. 'bayes_opt' allows this and not 'hyperopt'.
'''

