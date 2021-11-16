

import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split


"""
Titanic (Smaller) Dataset with Bayesian Hyper-Parameters Optimization

A binary classification problem. Target variable is 'Survived'
"""


# Read CSV file-
titanic_data = pd.read_csv("Titanic_Dataset.csv")

# Check dimension of dataset-
titanic_data.shape
# (2201, 4)

# Check for missing values-
titanic_data.isnull().values.any()
# False

titanic_data.isnull().sum().sum()
# 0

titanic_data.columns
# Index(['Class', 'Sex', 'Age', 'Survived'], dtype='object')


# Get distribution of target variable 'Survived'-
titanic_data['Survived'].value_counts()
'''
No     1490
Yes     711
Name: Survived, dtype: int64
'''

titanic_data.dtypes
'''
Class       object
Sex         object
Age         object
Survived    object
dtype: object
'''

# Python3 dict to contain label encoders for each column-
le_d = {}

for col in titanic_data.columns:
    le_d[col] = preprocessing.LabelEncoder().fit(titanic_data[col])

le_d.keys()
# dict_keys(['Class', 'Sex', 'Age', 'Survived'])

le_d['Class']
# LabelEncoder()

for col in titanic_data.columns:
    titanic_data[col] = le_d[col].transform(titanic_data[col])

# Sanity check-
titanic_data.head()
'''
   Class  Sex  Age  Survived
0      2    1    1         0
1      2    1    1         0
2      2    1    1         0
3      2    1    1         0
4      2    1    1         0
'''


# Split dataset into features and labels-
X = titanic_data.drop('Survived', axis = 1)

X.columns
# Index(['Class', 'Sex', 'Age'], dtype='object')

y = titanic_data['Survived']

X.shape, y.shape
# ((2201, 3), (2201,))

# Create training and testing datasets-
X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size = 0.3, stratify = y)

X_train.shape, y_train.shape
# ((1540, 3), (1540,))

X_test.shape, y_test.shape
# ((661, 3), (661,))




from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score


# LogisticRegression model-
log_model = LogisticRegression()
log_model.fit(X_train, y_train)

y_pred = log_model.predict(X_test)

acc_logistic = accuracy_score(y_true = y_test, y_pred = y_pred)
prec_logistic = precision_score(y_true = y_test, y_pred = y_pred)
rec_logistic = recall_score(y_true = y_test, y_pred = y_pred)

print("\nBase LogisticRegression model metrics are:")
print(f"accuracy = {acc_logistic * 100:.3f}%, precision = "
        f"{prec_logistic * 100:.3f}% & recall = "
        f"{rec_logistic * 100:.3f}%")
# Base LogisticRegression model metrics are:
# accuracy = 78.517%, precision = 75.000% & recall = 50.467% 


# SGDClassifier model-
sgd_model = SGDClassifier()
sgd_model.fit(X_train, y_train)

y_pred_sgd = sgd_model.predict(X_test)

acc_sgd = accuracy_score(y_true = y_test, y_pred = y_pred_sgd)
prec_sgd = precision_score(y_true = y_test, y_pred = y_pred_sgd)
rec_sgd = recall_score(y_true = y_test, y_pred = y_pred_sgd)

print("\nBase SGD model metrics are:")
print(f"accuracy = {acc_sgd * 100:.3f}%, precision = "
        f"{prec_sgd * 100:.3f}% & recall = "
        f"{rec_sgd * 100:.3f}%")
# Base SGD model metrics are:
# accuracy = 41.755%, precision = 34.369% & recall = 87.850% 




# DecisionTree classifier model-
dtc_model = DecisionTreeClassifier()
dtc_model.fit(X_train, y_train)

y_pred_dt = dtc_model.predict(X_test)

acc_dt = accuracy_score(y_true = y_test, y_pred = y_pred_dt)
prec_dt = precision_score(y_true = y_test, y_pred = y_pred_dt)
rec_dt = recall_score(y_true = y_test, y_pred = y_pred_dt)

print("\nBase DecisionTree classifier model metrics are:")
print(f"accuracy = {acc_dt * 100:.3f}%, precision = "
        f"{prec_dt * 100:.3f}% & recall = "
        f"{rec_dt * 100:.3f}%")
# Base DecisionTree classifier model metrics are:
# accuracy = 78.669%, precision = 91.954% & recall = 37.383% 




# RandomForest classifier model-
rfc_model = RandomForestClassifier(n_estimators = 200)
rfc_model.fit(X_train, y_train)

y_pred_rfc = rfc_model.predict(X_test)

acc_rfc = accuracy_score(y_true = y_test, y_pred = y_pred_rfc)
prec_rfc = precision_score(y_true = y_test, y_pred = y_pred_rfc)
rec_rfc = recall_score(y_true = y_test, y_pred = y_pred_rfc)

print("\nbase randomforest (200 trees) classifier model metrics are:")
print(f"accuracy = {acc_rfc * 100:.3f}%, precision = "
        f"{prec_rfc * 100:.3f}% & recall = "
        f"{rec_rfc * 100:.3f}%")
# Base RandomForest (200 trees) classifier model metrics are:
# accuracy = 78.669%, precision = 91.954% & recall = 37.383%




'''
Bayesian Optimization code

https://www.analyticsvidhya.com/blog/2021/05/bayesian-optimization-bayes_opt-or-hyperopt/
'''


from hyperopt import hp, fmin, tpe
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer


# Specify hyper-paramters to be tuned-
hyperparams = {
        'n_estimators': hp.randint('n_estimators', 100, 800),
        'criterion': hp.choice('criterion', ('gini', 'entropy')),
        'max_depth': hp.randint('max_depth', 2, 10),
        'min_samples_split': hp.uniform('min_samples_split', 0.1, 0.5)
        }


# Convert accuracy metric to a scorer-
acc_score = make_scorer(accuracy_score)


def objective_fn(params):
    params = {
            'n_estimators': params['n_estimators'],
            'criterion': params['criterion'],
            'max_depth': params['max_depth'],
            'min_samples_split': params['min_samples_split']
            }
    rfc_model = RandomForestClassifier(**params)
    best_score = cross_val_score(
            rfc_model, X_train,
            y_train, scoring = acc_score,
            cv = 3).mean()
    return 1 - best_score


rfc_best_params = fmin(
        fn = objective_fn, space = hyperparams,
        max_evals = 50, algo = tpe.suggest
        )

# 100%|████████| 50/50 [01:46<00:00,  2.14s/trial, best loss: 0.21558670418660864]

# best loss achieved = 0.216 which means that best accuracy = 1 – 0.216 = 0.784

rfc_best_params
'''
{'criterion': 0,
 'max_depth': 6,
 'min_samples_split': 0.10666750337016682,
 'n_estimators': 343}
'''


# These set of hyper-parameters can be further tuned with RandomizedSearchCV,
# GridSearchCV, etc.


# Sanity check- Use 'best' params to train a new model and then, check it's
# metrics-
best_rfc_model = RandomForestClassifier(
        criterion = 'gini', max_depth = 6,
        min_samples_split = 0.11, n_estimators = 343)
best_rfc_model.fit(X_train, y_train)

y_pred_best = best_rfc_model.predict(X_test)

acc_best = accuracy_score(y_true = y_test, y_pred = y_pred_best)
prec_best = precision_score(y_true = y_test, y_pred = y_pred_best)
rec_best = recall_score(y_true = y_test, y_pred = y_pred_best)

print("\n'best' randomforest classifier model metrics are:")
print(f"accuracy = {acc_best * 100:.3f}%, precision = "
        f"{prec_best * 100:.3f}% & recall = "
        f"{rec_best * 100:.3f}%")
# 'best' randomforest classifier model metrics are:
# accuracy = 78.366%, precision = 85.859% & recall = 39.720%



