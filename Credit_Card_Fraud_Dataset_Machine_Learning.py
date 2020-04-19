

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE

from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score


'''
Credit Card Fraud Detection
Anonymized credit card transactions labeled as fraudulent or genuine


Fore more information, refer-
https://www.kaggle.com/mlg-ulb/creditcardfraud/data
'''




# Read in CSV dataset-
data = pd.read_csv("creditcard.csv")

# Check shape/dimension-
data.shape
# (284807, 31)

# Check for missing values-
data.isnull().sum().sum()
# 0

data.isnull().values.any()
# False

data.isnull().values.sum()
# 0


# Get distribution of target attribute-
data.Class.value_counts()
# OR-
# data['Class'].value_counts()
'''
0    284315
1       492
Name: Class, dtype: int64
'''

# NOTE:
# There is a huge imbalance in the dataset. Only about
# 0.173% of the data has a fraud!




# Visualize target attribute distribution-
sns.countplot('Class', data = data)

plt.show("Target attribute Visualization")
plt.show()


# Plot data distribution of dataset using boxplots-
sns.boxplot(data=data)

plt.xticks(rotation = 20)
plt.xlabel("Attributes")
plt.ylabel("Count")
plt.title("Dataset distribution using Boxplots")
plt.show()


# Visualize heatmap for correlation between different attributes-
data_corr = data.corr()

sns.heatmap(data=data_corr)

plt.title("Heatmap: fraud detection dataset")
plt.show()




# Split dataset into features (X) and target (y)-
X = data.drop('Class', axis = 1)
y = data['Class']


# Scale features (X)-
std_scaler = StandardScaler()
X_scaled = std_scaler.fit_transform(X)


# Feature Extraction using PCA-

# Extract to 10 features (X)-
pca = PCA(n_components=10)

# Train PCA on features (X) and perform dimensionality reduction-
pca_features_extracted = pca.fit_transform(X_scaled)

# Convert results of PCA to Pandas DataFrame-
X_pca = pd.DataFrame(data=pca_features_extracted)

# Get shape/dimension-
X_pca.shape
# (284807, 10)


# Split features and target into training and testing datasets-
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, stratify = y)

print("\nShapes of training and testing sets are:")
print("X_train.shape = {0}, y_train.shape = {1}".format(X_train.shape, y_train.shape))
print("X_test.shape = {0}, y_test.shape = {1}\n".format(X_test.shape, y_test.shape))
# Shapes of training and testing sets are:
# X_train.shape = (199364, 30), y_train.shape = (199364,)
# X_test.shape = (85443, 30), y_test.shape = (85443,)


# Get distribution of target attribute for training and testing sets-
y_train.value_counts()
'''
0    199020
1       344
Name: Class, dtype: int64
'''

y_test.value_counts()
'''
0    85295
1      148
Name: Class, dtype: int64
'''




# SMOTE - Synthetic Minority Over-sampling Technique
# Experiments with 'SMOTE' to balance the dataset-
smt = SMOTE()

X_bal, y_bal = smt.fit_resample(X_scaled, y)

y_bal.value_counts()
'''
1    284315
0    284315
Name: Class, dtype: int64
'''

y.value_counts()
'''
0    284315
1       492
Name: Class, dtype: int64
'''

X_bal.shape, X.shape
# ((568630, 30), (284807, 30))

# Create new training and testing sets uing 'X_bal' and 'y_bal'
X_train_bal, X_test_bal, y_train_bal, y_test_bal = train_test_split(X_bal, y_bal, test_size=0.3, stratify = y_bal)

print("\nShapes of training and testing sets using SMOTE are:")
print("X_train.shape = {0}, y_train.shape = {1}".format(X_train_bal.shape, y_train_bal.shape))
print("X_test.shape = {0}, y_test.shape = {1}\n".format(X_test_bal.shape, y_test_bal.shape))
#



# Experiment with different Machine Learning classifiers:


# Logistic Regression classifier-

# Initialize a new model-
log_clf = LogisticRegression()

# Train model on training data-
log_clf.fit(X_train, y_train)

# Make predictions using trained model-
y_pred_log = log_clf.predict(X_test)

# Get model metrics on validation data-
accuracy_log = accuracy_score(y_test, y_pred_log)
precision_log = precision_score(y_test, y_pred_log)
recall_log = recall_score(y_test, y_pred_log)

print("\nLogistic Regression (base) model metrics are:")
print("Accuracy = {0:.4f}, Precision = {1:.4f} and Recall = {2:.4f}\n".format(
	accuracy_log, precision_log, recall_log
	))
# Logistic Regression (base) model metrics are:
# Accuracy = 0.9991, Precision = 0.8304 and Recall = 0.6284


# LightGBM classifier-

lgb_clf = lgb.LGBMClassifier()

lgb_clf.fit(X_train, y_train)

y_pred_lgb = lgb_clf.predict(X_test)

accuracy_lgb = accuracy_score(y_test, y_pred_lgb)
precision_lgb = precision_score(y_test, y_pred_lgb)
recall_lgb = recall_score(y_test, y_pred_lgb)

print("\nLightGBM (base) model metrics are:")
print("Accuracy = {0:.4f}, Precision = {1:.4f} and Recall = {2:.4f}\n".format(
	accuracy_lgb, precision_lgb, recall_lgb
	))
# LightGBM (base) model metrics are:
# Accuracy = 0.9938, Precision = 0.1492 and Recall = 0.5473


# XGBoost classifier-

xgb_clf = xgb.XGBClassifier()

xgb_clf.fit(X_train, y_train)

y_pred_xgb = xgb_clf.predict(X_test)

accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
precision_xgb = precision_score(y_test, y_pred_xgb)
recall_xgb = recall_score(y_test, y_pred_xgb)

print("\nXGBoost (base) model metrics are:")
print("Accuracy = {0:.4f}, Precision = {1:.4f} and Recall = {2:.4f}\n".format(
	accuracy_xgb, precision_xgb, recall_xgb
	))
# XGBoost (base) model metrics are:
# Accuracy = 0.9994, Precision = 0.9244 and Recall = 0.7432


# Random Forest classifier-

rf_clf = RandomForestClassifier(n_estimators=100)

rf_clf.fit(X_train, y_train)

y_pred_rf = rf_clf.predict(X_test)

accuracy_rf = accuracy_score(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf)
recall_rf = recall_score(y_test, y_pred_rf)

print("\nRandom Forest (# of trees = 100) (base) model metrics are:")
print("Accuracy = {0:.4f}, Precision = {1:.4f} and Recall = {2:.4f}\n".format(
	accuracy_rf, precision_rf, recall_rf
	))
# Random Forest (# of trees = 100) (base) model metrics are:
# Accuracy = 0.9995, Precision = 0.9180 and Recall = 0.7568



# NEXT STEPS:

# 1. Experiment with SMOTE datasets: X_train_bal to see whether there is any improvement?
# 2. Hyperparameter tuning for top performing models (XGBoost and Random Forest classifiers)
