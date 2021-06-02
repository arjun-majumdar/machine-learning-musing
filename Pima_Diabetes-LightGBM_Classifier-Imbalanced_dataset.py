

import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import lightgbm as lgb
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.model_selection import cross_val_score
import time
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform


'''
Predict for attribute- 'Outcome'

Refer-
https://medium.com/district-data-labs/parameter-tuning-with-hyperopt-faa86acdfdce

https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html
https://lightgbm.readthedocs.io/en/latest/Parameters.html
https://github.com/microsoft/LightGBM/tree/master/examples/python-guide
'''


# Read in CSV data-
pima_data = pd.read_csv("Pima_Diabetes_Data.csv")

# Dimension of data-
pima_data.shape
# (768, 9)

# Check for missing values-
pima_data.isnull().values.any()
# False

pima_data.isnull().sum().sum()
# 0

pima_data.dtypes
'''
Pregnancies                   int64
Glucose                       int64
BloodPressure                 int64
SkinThickness                 int64
Insulin                       int64
BMI                         float64
DiabetesPedigreeFunction    float64
Age                           int64
Outcome                       int64
dtype: object
'''


# Get distribution of target variable 'Outcome'-
pima_data['Outcome'].value_counts()
'''
0    500
1    268
Name: Outcome, dtype: int64
'''
# NOTE: target attribute is imbalanced!


'''
# Visualize distribution of target variable 'Outcome'-

# Using 'seaborn'-
sns.distplot(pima_data["Outcome"], kde=False, rug=True)

plt.title("Target Variable- 'Outcome' Distribution")
plt.show()


# Using 'matplotlib'-
plt.hist(pima_data["Outcome"])

# Using 'seaborn'-
sns.boxplot(data = pima_data)

plt.title("Target Variable- 'Outcome' Distribution")
plt.xlabel("'Outcome' variable")
plt.ylabel("Count")
plt.show()
'''

'''
# Visualize heat map/correlogram of attributes in dataset-
# Calculate correlation matrix of 'pima_data'-
pima_corr = pima_data.corr()

# Create heat map-
sns.heatmap(pima_corr)

plt.xticks(rotation = 30)
plt.yticks(rotation = 30)
plt.title("Heat Map of attributes in dataset")
plt.show()
'''




# Split dataset into features (X) and label (y)-
X = pima_data.drop('Outcome', axis = 1)
y = pima_data['Outcome']

# Convert all numeric values to float-
X = X.values.astype("float")

# Convert 'X' from numpy array to pandas DataFrame-
X = pd.DataFrame(X, columns=pima_data.columns.tolist()[:-1])


# Visualize heat map/correlogram of features

# Normalize/Scale dataset-
# mm_scaler = MinMaxScaler()
rb_scaler = RobustScaler()

# X_scaled = mm_scaler.fit_transform(X)
X_scaled = rb_scaler.fit_transform(X)

# Convert 'X_scaled' from numpy array to pandas DataFrame-
X_scaled = pd.DataFrame(X_scaled, columns=X.columns.tolist())


# Divide attributes & labels into training & testing sets-
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.30, stratify = y)

print("\nDimensions of training and testing sets are:")
print("X_train = {0}, y_train = {1}, X_test = {2} and y_test = {3}\n\n".format(X_train.shape, y_train.shape, X_test.shape, y_test.shape))





# Using LightGBM classifier-

# Initialize a base LightGBM classifier-
lgbm_clf = lgb.LGBMClassifier()

# Train model using 5-fold cross validation-
cvs = cross_val_score(lgbm_clf, X_train, y_train, cv = 5, scoring = 'accuracy')
print("\nLightGBM base classifier using 5-fold CV 'accuracy' mean metric = {0:.4f}\n\n".format(cvs.mean()))
# LightGBM base classifier using 5-fold CV 'accuracy' mean metric = 0.7634

print("\nLightGBM base classifier using 5-fold CV 'accuracy' standard deviation metric = {0:.4f}\n\n".format(cvs.std()))
# LightGBM base classifier using 5-fold CV 'accuracy' standard deviation metric = 0.0305

# Train base model on training data-
lgbm_clf.fit(X_train, y_train)

# Make predictions using base model-
y_pred = lgbm_clf.predict(X_test)


# Get model metrics-
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1score = f1_score(y_test, y_pred)

print("\nLightGBM base model metrics are:")
print("Accuracy = {0:.4f}, Precision = {1:.4f}, Recall = {2:.4f} & F1-Score = {3:.4f}".format(accuracy, precision, recall, f1score))
# LightGBM base model metrics are:
# Accuracy = 0.7273, Precision = 0.6184, Recall = 0.5802 & F1-Score = 0.5987

# Warning:
# [LightGBM] [Warning] No further splits with positive gain, best gain: -inf




# Parameters to be used by LightGBM classifier-
params = {
    'boosting_type': 'gbdt',
    # 'objective': 'multiclass',
    'objective': 'binary',
    # 'num_class': 11,
    # 'num_class': 2,
    # 'metric': 'multi_logloss',
    'metric': 'binary_logloss',
    'is_unbalance': True
}


# Create dataset for lightgbm
# If you want to re-use data, remember to set free_raw_data=False
lgb_train = lgb.Dataset(X_train, y_train, free_raw_data=False)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train, free_raw_data=False)

# https://www.kaggle.com/aloisiodn/lgbm-starter-early-stopping-0-9539
# G: lightgbm early stopping example
# categorical = ['Pregnancies']

# Initialize and train LightGBM classifier on training data-
# lgb_clf = lgb.LGBMClassifier(objective: 'multiclass', num_class:3, is_unbalance = True)
# lgb_clf = lgb.train(params, lgb_train, num_boost_round=10, valid_sets=lgb_train)
# lgb_clf = lgb.train(params, lgb_train, valid_sets=lgb_train, verbose_eval = True,
# 	early_stopping_rounds = 20, num_boost_round = 100, categorical_feature = categorical)
lgb_clf = lgb.train(params, lgb_train, valid_sets=lgb_train, verbose_eval = True,
 	early_stopping_rounds = 20, num_boost_round = 100)
# lgb_clf.fit(X_train, y_train)

# Make predictions using trained model-
y_pred = lgb_clf.predict(X_test)
# OR-
# y_pred = lgb_clf.predict(X_test, num_iteration=lgb_clf.best_iteration)
    
# Convert multi-label into single label-
# labels = np.argmax(y_pred, axis = -1)
labels = [1 if x >= 0.5 else 0 for x in y_pred]
    
# Get model metrics on testing data-
accuracy = accuracy_score(y_test, labels)
precision = precision_score(y_test, labels, average = 'macro')
recall = recall_score(y_test, labels, average = 'macro')

print("\nLightGBM parameterized model metrics are:")
print("Accuracy = {0:.4f}, Precision = {1:.4f} and Recall = {2:.4f}\n".format(accuracy, precision, recall))
# LightGBM parameterized model metrics are:
# Accuracy = 0.7273, Precision = 0.6995 and Recall = 0.6935




# LightGBM Cross-Validation-
lgb_cv = lgb.cv(params = params, train_set = lgb_train, num_boost_round = 100,
	nfold = 5, stratified = True, early_stopping_rounds=None, verbose_eval=None)


type(lgb_cv)
# dict

lgb_cv.keys()
# dict_keys(['binary_logloss-mean', 'binary_logloss-stdv'])




# Initialize a LightGBM classifier using parameters-
lgb_clf = lgb.LGBMClassifier(boosting_type='gbdt', objective='binary',
	metric = 'binary_logloss', is_unbalance = True)
# metric = 'multi_logloss'
# objective = 'multiclass'
# num_class = 2

# Train model on training data-
lgb_clf.fit(X_train, y_train)

# Make predictions using trained model-
y_pred = lgb_clf.predict(X_test)
    
# Convert multi-label into single label-
# labels = np.argmax(y_pred, axis = -1)
    
# Get model metrics on testing data-
# accuracy = accuracy_score(y_test, labels)
accuracy = accuracy_score(y_test, y_pred)

# precision = precision_score(y_test, labels, average = 'macro')
precision = precision_score(y_test, y_pred)

# recall = recall_score(y_test, labels, average = 'macro')
recall = recall_score(y_test, y_pred)

print("\nLightGBM parameterized model metrics are:")
print("Accuracy = {0:.4f}, Precision = {1:.4f} and Recall = {2:.4f}\n".format(accuracy, precision, recall))
# LightGBM parameterized model metrics are:
# Accuracy = 0.7186, Precision = 0.6991 and Recall = 0.7123




# Use 5-fold CV for metrics-
cvs_parameterized_lgb = cross_val_score(lgb_clf, X_train, y_train, cv = 5, scoring = 'accuracy')
print("\n5-fold CV LightGBM classifier metrics for accuracy are:")
print("Mean = {0:.4f} & Std. dev = {1:.4f}".format(cvs_parameterized_lgb.mean(), cvs_parameterized_lgb.std()))
# 5-fold CV LightGBM classifier metrics for accuracy are:
# Mean = 0.7430 & Std. dev = 0.0255

cvs_parameterized_lgb = cross_val_score(lgb_clf, X_train, y_train, cv = 5, scoring = 'precision')
print("\n5-fold CV LightGBM classifier metrics for precision are:")
print("Mean = {0:.4f} & Std. dev = {1:.4f}".format(cvs_parameterized_lgb.mean(), cvs_parameterized_lgb.std()))
# 5-fold CV LightGBM classifier metrics for precision are:
# Mean = 0.6352 & Std. dev = 0.0371

cvs_parameterized_lgb = cross_val_score(lgb_clf, X_train, y_train, cv = 5, scoring = 'recall')
print("\n5-fold CV LightGBM classifier metrics for recall are:")
print("Mean = {0:.4f} & Std. dev = {1:.4f}".format(cvs_parameterized_lgb.mean(), cvs_parameterized_lgb.std()))
# 5-fold CV LightGBM classifier metrics for recall are:
# Mean = 0.6154 & Std. dev = 0.0647



plt.plot(cvs_parameterized_lgb)
plt.xlabel("Cross-Validation fold")
plt.ylabel("Recall score")
# plt.xlim(1, 6)
plt.title("5-fold CV for 'recall' metric")
plt.show()

