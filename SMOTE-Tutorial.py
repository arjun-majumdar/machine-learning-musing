

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import sklearn.svm


"""
SMOTE (Synthetic Minority Over-sampling Technique)

SMOTE is an over-sampling method.
What it does is, it creates synthetic (not duplicate) samples of the minority
class. Hence, it makes the minority class equal to the majority class.
SMOTE does this by selecting similar records and altering that record one
column at a time by a random amount within the difference to the neighbouring
records.

"""


# Read in CSV file-
bank_data = pd.read_csv("bank-additional-full.csv", sep = ';')

# Get dimension of dataset-
bank_data.shape
# (41188, 21)

# Check for missing values-
bank_data.isnull().values.any()
# False

# OR-
bank_data.isnull().sum().sum()
# 0



# Filter non-numeric attributes from dataset-
non_numeric_attributes = bank_data.select_dtypes(exclude=[np.number]).columns.tolist()

non_numeric_attributes
'''
['job',
 'marital',
 'education',
 'default',
 'housing',
 'loan',
 'contact',
 'month',
 'day_of_week',
 'poutcome',
 'y']
'''


# Filter numeric attributes from dataset-
numeric_attributes = bank_data.select_dtypes(include=[np.number]).columns.tolist()

numeric_attributes
'''
['age',
 'duration',
 'campaign',
 'pdays',
 'previous',
 'emp.var.rate',
 'cons.price.idx',
 'cons.conf.idx',
 'euribor3m',
 'nr.employed']
'''


"""
# Visualize distribution of dataset using boxplots-
fig=plt.figure(figsize=(10, 9), dpi= 80, facecolor='w', edgecolor='k')

sns.boxplot(data=bank_data)
plt.xticks(rotation = 30)
plt.title("Distribution of dataset - Boxplots")
plt.show()


# Visualize heatmap/correlogram-
# Compute correlation matrix for numeric attributes in dataset-
bank_data_corr = bank_data.loc[:, numeric_attributes].corr()

sns.heatmap(data = bank_data_corr, annot=True)
plt.xticks(rotation = 30)
plt.title("Correlogram of numeric attributes - Ban Dataset")
plt.show()


# To visualize distribution of 'nr.employed' attribute-
sns.distplot(bank_data['nr.employed'], kde=True)
plt.title("Distribution of 'nr.employed' attribute")
plt.show()
"""


# Perform one-hot encoding for non-numeric attributes in dataset-
bank_data_dummies = pd.get_dummies(bank_data.loc[:, non_numeric_attributes])

# Add numeric attributes-
bank_data_dummies = pd.concat([bank_data.loc[:, numeric_attributes], bank_data_dummies], axis = 1)

bank_data_dummies.drop(['y_no', 'y_yes'], axis = 1, inplace=True)
bank_data_dummies['y'] = bank_data['y']


"""
# Scale numeric attributes-
std_scaler = StandardScaler()

bank_data_scaled = std_scaler.fit_transform(bank_data_dummies.loc[:, numeric_attributes])

type(bank_data_scaled)
# numpy.ndarray

# Convert from numpy array to Pandas DataFrame-
bank_data_scaled = pd.DataFrame(bank_data_scaled, columns=numeric_attributes)
"""


# Divide dataset into features (X) and label (y)-
X = bank_data_dummies.drop('y', axis = 1)
y = bank_data['y']


# Visualize distribution of target-
y.value_counts()
'''
no     36548
yes     4640
Name: y, dtype: int64
'''

'''
# Visualize distribution of target-
y.value_counts().plot(kind = 'bar')
plt.xticks(rotation = 30)
plt.title("Distribution of target attribute 'y'")
plt.show()
'''


# Do label encoding for 'y'-
# Initialize label encoder object-
le = LabelEncoder()

# Perform label encoding-
y = le.fit_transform(y)

# To reverse label encoding-
# inv_y = le.inverse_transform(y)


# Get distribution of labels within 'y'-
np.bincount(y)
# array([36548,  4640])


# Divide features (X) and label (y) into training and testing sets-
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, stratify = y)

print("\nDimensions of training and testing sets are:")
print("X_train = {0}, y_train = {1}, X_test = {2} and y_test = {3}\n".format(X_train.shape, y_train.shape, X_test.shape, y_test.shape))
# Dimensions of training and testing sets are:
# X_train = (28831, 63), y_train = (12357, 63), X_test = (28831,) and y_test = (12357,)


# Give weights to the two classes of target attribute-
wts = np.array(36548 / 41188, 4640 / 41188)





# Using a DecisionTree classifier-

# Initialize a DT classifier-
dt_clf = DecisionTreeClassifier()

# Train model on training data-
dt_clf.fit(X_train, y_train)

# Make predictions using trained model-
y_pred = dt_clf.predict(X_test)

# Count occurrences of 'yes' and 'no' in 'y_pred'-
unique, counts = np.unique(y_pred, return_counts=True)

# Create a dictionary object-
count_occurrence_y_pred = dict(zip(unique, counts))

count_occurrence_y_pred
# {'no': 10962, 'yes': 1395}


# Get model metrics-
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
# recall = recall_score(y_test, y_pred, pos_label="yes")
recall = recall_score(y_test, y_pred)

print("\nDecision Tree (base) model metrics are:")
print("Accuracy = {0:.4f}, Precision = {1:.4f} and Recall = {2:.4f}\n".format(accuracy, precision, recall))
# Decision Tree (base) model metrics are:
# Accuracy = 0.8905, Precision = 0.5140 and Recall = 0.5151

print("\nConfusion Matrix of (base) DT model is:\n{0}\n".format(confusion_matrix(y_test, y_pred)))
'''
Confusion Matrix of (base) DT model is:
[[10287   678]
 [  675   717]]
'''




# Using SMOTE:

# Initialize SMOTE object-
smt = SMOTE()

# Get training and testing sets using SMOTE-
X_train, y_train = smt.fit_sample(X_train, y_train)

X_train.shape, y_train.shape
# ((51166, 63), (51166,))

type(X_train), type(y_train)
# (numpy.ndarray, numpy.ndarray)

# Check the number of classes in 'y_train'-
np.bincount(y_train)
# array([25583, 25583])




"""
NearMiss:

NearMiss is an under-sampling technique.
Instead of resampling the Minority class, using a distance, this will make
the majority class equal to minority class.
"""

# Initialize NearMiss object-
near_miss = NearMiss()

# Get training sets-
X_train, y_train = near_miss.fit_sample(X_train, y_train)

X_train.shape, y_train.shape
# ((6496, 63), (6496,))

type(X_train), type(y_train)
# (numpy.ndarray, numpy.ndarray)

# Check the number of classes in 'y_train'-
np.bincount(y_train)
# array([3248, 3248])




# Using Decision Tree classifier on SMOTE generated synthetic datasets-
dt_clf2 = DecisionTreeClassifier()

# Train DT on training data-
dt_clf2.fit(X_train, y_train)

# Make predictions using trained model-
y_pred2 = dt_clf2.predict(X_test)

# Get model metrics-
accuracy = accuracy_score(y_test, y_pred2)
precision = precision_score(y_test, y_pred2)
recall = recall_score(y_test, y_pred2)

print("\nDecision Tree (base) model metrics using SMOTE are:")
print("Accuracy = {0:.4f}, Precision = {1:.4f} and Recall = {2:.4f}\n".format(accuracy, precision, recall))
# Decision Tree (base) model metrics using SMOTE are:
# Accuracy = 0.8905, Precision = 0.5135 and Recall = 0.5323

print("\nConfusion Matrix of (base) DT model is:\n{0}\n".format(confusion_matrix(y_test, y_pred2)))
'''
Confusion Matrix of (base) DT model is:
[[10263   702]
 [  651   741]]
'''




# Using XGBoost classifier-

# Initiliaze base classifier-
# weight (list or numpy 1-D array , optional) –
# Weight for each instance.
# xgb_clf = xgb.XGBClassifier(weight = wts)
xgb_clf = xgb.XGBClassifier(scale_pos_weight=36548/4640)
'''
scale_pos_weight [default=1]-
Control the balance of positive and negative weights, useful for unbalanced classes.
A typical value to consider: sum(negative instances) / sum(positive instances).
'''

# Train model on training data-
xgb_clf.fit(X_train, y_train)

# Make predictions using trained model-
y_pred_xgb = xgb_clf.predict(X_test)

# Get model metrics-
accuracy = accuracy_score(y_test, y_pred_xgb)
# precision = precision_score(y_test, y_pred_xgb, pos_label="yes")
precision = precision_score(y_test, y_pred_xgb)
recall = recall_score(y_test, y_pred_xgb)

# print("\nXGBoost classifier (base) model with 'weight' parameter metrics are:")
print("\nXGBoost classifier (base) model with 'scale_pos_weight' parameter metrics are:")
print("Accuracy = {0:.4f}, Precision = {1:.4f} and Recall = {2:.4f}\n".format(accuracy, precision, recall))
# XGBoost classifier (base) model with 'weight' parameter metrics are:
# Accuracy = 0.9115, Precision = 0.6346 and Recall = 0.5065
# XGBoost classifier (base) model with 'scale_pos_weight' parameter metrics are:
# Accuracy = 0.8580, Precision = 0.4391 and Recall = 0.9397

print("\nConfusion matrix is:\n{0}\n".format(confusion_matrix(y_test, y_pred_xgb)))
'''
Confusion matrix is:
[[9294 1671]
 [  84 1308]]
'''


# Use 5-fold CV with XGBoost classifier-
xgb_cvs = cross_val_score(xgb.XGBClassifier(weight = wts), X_train, y_train, cv = 5, scoring = 'precision')

print("\nMetrics of using 5-fold CV for XGBoost classifier using 'precision' scoring are:")
print("Mean = {0:.4f} and Standard Deviation = {1:.4f}\n".format(xgb_cvs.mean(), xgb_cvs.std()))
# Metrics of using 5-fold CV for XGBoost classifier using 'precision' scoring are:
# Mean = 0.6689 and Standard Deviation = 0.0205

xgb_cvs = cross_val_score(xgb.XGBClassifier(weight = wts), X_train, y_train, cv = 5, scoring = 'recall')

print("\nMetrics of using 5-fold CV for XGBoost classifier using 'recall' scoring are:")
print("Mean = {0:.4f} and Standard Deviation = {1:.4f}\n".format(xgb_cvs.mean(), xgb_cvs.std()))
# Metrics of using 5-fold CV for XGBoost classifier using 'recall' scoring are:
# Mean = 0.5179 and Standard Deviation = 0.0090

xgb_cvs = cross_val_score(xgb.XGBClassifier(weight = wts), X_train, y_train, cv = 5, scoring = 'accuracy')

print("\nMetrics of using 5-fold CV for XGBoost classifier using 'accuracy' scoring are:")
print("Mean = {0:.4f} and Standard Deviation = {1:.4f}\n".format(xgb_cvs.mean(), xgb_cvs.std()))
# Metrics of using 5-fold CV for XGBoost classifier using 'accuracy' scoring are:
# Mean = 0.9167 and Standard Deviation = 0.0028


# Use XGBoost classifier without 'weight' parameter-
xgb_clf2 = xgb.XGBClassifier()

# Train model on training data-
xgb_clf2.fit(X_train, y_train)

# Make predictions using trained model-
y_pred_xgb2 = xgb_clf2.predict(X_test)

# Get model metrics-
accuracy = accuracy_score(y_test, y_pred_xgb2)
precision = precision_score(y_test, y_pred_xgb2)
recall = recall_score(y_test, y_pred_xgb2)

print("\nXGBoost classifier (base) model without 'weight' parameter metrics are:")
print("Accuracy = {0:.4f}, Precision = {1:.4f} and Recall = {2:.4f}\n".format(accuracy, precision, recall))
# XGBoost classifier (base) model without 'weight' parameter metrics are:
# Accuracy = 0.9115, Precision = 0.6346 and Recall = 0.5065




# Using LightGBM classifier-

# Initialize classifier-
lgb_clf = lgb.LGBMClassifier(is_unbalance = True)

# Train classifier on training data-
lgb_clf.fit(X_train, y_train)

# Make predictions using trained model-
y_pred_lgb = lgb_clf.predict(X_test)

# Get model metrics using trained model-
accuracy = accuracy_score(y_test, y_pred_lgb)
precision = precision_score(y_test, y_pred_lgb)
recall = recall_score(y_test, y_pred_lgb)

print("\nLightGBM classifier (base) model metrics are:")
print("Accuracy = {0:.4f}, Precision = {1:.4f} and Recall = {2:.4f}\n".format(accuracy, precision, recall))
# LightGBM classifier (base) model metrics are:
# Accuracy = 0.8751, Precision = 0.4720 and Recall = 0.9188

print("\nConfusion Matrix of (base) LightGBM model is:\n{0}\n".format(confusion_matrix(y_test, y_pred_lgb)))
'''
Confusion Matrix of (base) LightGBM model is:
[[9534 1431]
 [ 113 1279]]
'''



