

import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import RandomForestClassifier 
from sklearn.pipeline import Pipeline
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
import time


'''
Have to predict for target attribute- 'class'
'''


start_time = time.time()

# Read-in data-
breast_cancer_data = pd.read_csv("Breast_Cancer_Training_Data.csv")

# Get shape of data-
breast_cancer_data.shape
# (285, 32)

# Drop 'ID' attribute-
breast_cancer_data.drop("ID", axis = 1, inplace=True)

# Get shape of data after dropping 'ID' column from data-
breast_cancer_data.shape
# (285, 31)

# Get names of attributes-
# breast_cancer_data.columns.tolist()


# Check for missing values-
# breast_cancer_data.isnull().any()	# No missing data!


# Get distribution of target attribute: 'class'-
breast_cancer_data['class'].value_counts()
"""
B    189
M     96
Name: class, dtype: int64

# NOTE: There is a class imbalance for target variable!
"""


'''
# Visualize data distribution for each attribute in dataset using boxplots-
sns.boxplot(data = breast_cancer_data)

plt.title("Visualizing data distribution for each attribute using boxplots")
plt.xticks(rotation = 20)
plt.show()

# From the boxplots, we can see that using StandardScaler or MinMaxScaler will NOT be
# right choice. Hence choosing RobustScaler
'''


# Divide data into features (X) and label (y)-

# Encode different values for 'class' attribute as integers-
encoder = LabelEncoder()
encoder.fit(breast_cancer_data['class'])

breast_cancer_data['Encoded_Class'] = encoder.transform(breast_cancer_data['class'])

# 'X' contains attributes-
X = breast_cancer_data.drop('class', axis = 1)

# 'y' contains labels-
y = breast_cancer_data['Encoded_Class']


# Divide features (X) and label (y) into training and testing sets-
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify = y)
# It’s necessary to use 'stratify' parameter as the labels are imbalanced.

print("\nDimensions of training and testing sets are:")
print("X_train = {0}, y_train = {1}, X_test = {2} & y_test = {3}\n\n".format(X_train.shape, y_train.shape, X_test.shape, y_test.shape))
# Dimensions of training and testing sets are:
# X_train = (228, 31), y_train = (228,), X_test = (57, 31) & y_test = (57,)




'''
As the name suggests, 'pipeline' class allows sticking multiple processes
into a single scikit-learn estimator.
'pipeline' class has fit, predict and score method just like any other
estimator (ex. LinearRegression).

To implement pipeline, as usual we have to separate features and labels from the
data-set first.

After having separated features and labels from dataset, we are ready to create a
pipeline object by providing with the list of steps.

Here our steps are robust scalar and random forest classifier.
These steps are list of tuples consisting of name and an instance of the
transformer or estimator.
'''
pipeline_steps = [('rb_scaler', RobustScaler()), ('rf_clf', RandomForestClassifier(n_estimators=30))]

# Define the pipeline object-
pipeline = Pipeline(pipeline_steps)

# Train base RF classifier on training data-
pipeline.fit(X_train, y_train)

# Make predictions using base RF classifier-
y_pred = pipeline.predict(X_test)


print("\nRandomForest classifier score on training data = {0:.4f}".format(pipeline.score(X_train, y_train)))
# RandomForest classifier score on training data = 1.0000

print("\nRandomForest classifier score on testing data = {0:.4f}\n".format(pipeline.score(X_test, y_test)))
# RandomForest classifier score on testing data = 0.9825


# Get base RF classifier model metrics-
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1score = f1_score(y_test, y_pred)

print("\nRandomForest base classifier model metrics are:")
print("Accuracy = {0:.4f}, Precision = {1:.4f}, Recall = {2:.4f} & F1-Score = {3:.4f}\n\n".format(accuracy, precision, recall, f1score))
# RandomForest base classifier model metrics are:
# Accuracy = 0.9825, Precision = 1.0000, Recall = 0.9474 & F1-Score = 0.9730




# Performing hyper parameter optimization-

# Using RandomizedSearchCV-
'''
Random Search is a similar approach to Grid Search however now instead of testing all
possible combinations, "n_iter" sets of parameters are randomly selected.

According to RandomSearchCV documentation, it is highly recommended to draw from continuous
distributions for continuous parameters, thus we will use uniform distribution U(0.6,1.0) for
subsample and column sampling by tree parameters.
Additionally, the number of estimators will be drawn from discrete uniform distribution.
'''

rs_params = {
		'rf_clf__n_estimators': sp_randint(30, 170),
		'rf_clf__max_features': ['auto', 'sqrt', 'log2'],
		'rf_clf__max_depth': [x for x in range(4, 14, 2)],
		'rf_clf__min_samples_split': [2, 5, 7, 10],
		'rf_clf__min_samples_leaf': [1, 2, 4, 6],
		'rf_clf__criterion': ['gini', 'entropy'],
		'rf_clf__bootstrap': [True, False]
}

print("\n\nPerforming RandomizedSearchCV using 10-fold CV now.\n\n")

# Initializing a RandomizedSearchCV object using 10-fold CV-
rs_cv = RandomizedSearchCV(pipeline, param_distributions=rs_params, n_iter=100, cv = 10)

# Train on training data-
rs_cv.fit(X_train, y_train)

print("\n\nBest parameters using RandomizedSearchCV are:\n{0}\n".format(rs_cv.best_params_))
'''
Best parameters using RandomizedSearchCV are:
{'rf_clf__bootstrap': True, 'rf_clf__criterion': 'gini', 'rf_clf__max_depth': 10,
'rf_clf__max_features': 'auto', 'rf_clf__min_samples_leaf': 1,
'rf_clf__min_samples_split': 5, 'rf_clf__n_estimators': 96}
'''

print("Best score achieved by RandomizedSearchCV = {0:.4f}\n\n".format(rs_cv.best_score_))
# Best score achieved by RandomizedSearchCV = 1.0000


# Finally, train a best RF classifier using parameters from above-
best_rf_clf = RandomForestClassifier(bootstrap=True, criterion='gini', max_depth=10, max_features='auto', min_samples_leaf=1, min_samples_split=5, n_estimators=96)

# Train best model on training data-
best_rf_clf.fit(X_train, y_train)

# Make predictions using best RF classifier-
y_pred_best = pipeline.predict(X_test)


# Get best RF classifier model metrics-
accuracy_best = accuracy_score(y_test, y_pred_best)
precision_best = precision_score(y_test, y_pred_best)
recall_best = recall_score(y_test, y_pred_best)
f1score_best = f1_score(y_test, y_pred_best)

print("\nRandomForest best classifier model metrics are:")
print("Accuracy = {0:.4f}, Precision = {1:.4f}, Recall = {2:.4f} & F1-Score = {3:.4f}\n\n".format(accuracy_best, precision_best, recall_best, f1score_best))
# RandomForest best classifier model metrics are:
# Accuracy = 1.0000, Precision = 1.0000, Recall = 1.0000 & F1-Score = 1.0000




end_time = time.time()
print("\n\nTotal time taken = {0:.4f} seconds \n\n".format(end_time - start_time))
