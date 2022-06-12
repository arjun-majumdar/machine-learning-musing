

"""
Local Cascade Ensemble (LCE) Tutorial


Local Cascade Ensemble (LCE) [Fauvel et al., 2022] is a new machine learning
method which proposes to answer the question: which one should we choose for
our given dataset, Random Forest (Breiman, 2001), or, XGBoost (Chen and Guestrin, 2016).

It combines their strengths and adopts a complementary diversification approach to
obtain a better generalizing predictor. Thus, LCE further enhances the prediction
performance of both Random Forest and XGBoost.


Reference-
https://towardsdatascience.com/random-forest-or-xgboost-it-is-time-to-explore-lce-2fed913eafb8
https://deeplearn.org/arxiv/130913/local-cascade-ensemble-for-multivariate-data-classification
https://lce.readthedocs.io/en/latest/
https://hal.inria.fr/hal-03599214/document
"""


from sklearn.datasets import load_iris, load_diabetes
from lce import LCEClassifier, LCERegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score, train_test_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Load Iris dataset-
iris_data = load_iris()

# Get train and test sets-
X_train, X_test, y_train, y_test = train_test_split(iris_data.data, iris_data.target)

X_train.shape, y_train.shape
# ((112, 4), (112,))

X_test.shape, y_test.shape
# ((38, 4), (38,))


# Initialize default LCE classifier-
lce_clf = LCEClassifier()

# Train model-
lce_clf.fit(X_train, y_train)

# Make predictions-
y_pred = lce_clf.predict(X_test)

# Compute metrics on test data-
acc = accuracy_score(y_true = y_test, y_pred = y_pred)
prec = precision_score(y_true = y_test, y_pred = y_pred, average = 'macro')
rec = recall_score(y_true = y_test, y_pred = y_pred, average = 'macro')

print("Default LCE classifier metrics on test dataset:")
print(f"accuracy = {acc * 100:.2f}%, precision = {prec * 100:.2f}% &"
f" recall = {rec * 100:.2f}%")
# Default LCE classifier metrics on test dataset:
# accuracy = 94.74%, precision = 94.89% & recall = 94.89%

print(f"Confusion matrix:\n{confusion_matrix(y_true = y_test, y_pred = y_pred)}")
'''
Confusion matrix:
[[11  0  0]
 [ 0 15  1]
 [ 0  1 10]]
'''


# Use 3-fold cross-validation using scikit model selection-
print(f"3-fold cross-validation score = {cross_val_score(lce_clf, iris_data.data, iris_data.target, cv = 3)}")
# 3-fold cross-validation score = [0.98 0.94 0.92]


# Use 20% missing values for each feature in dataset-
m = 0.2

for col in range(X_train.shape[1]):
    sub = np.random.choice(X_train.shape[0], int(X_train.shape[0] * m))
    X_train[sub, col] = np.nan


print(f"Total number of NaNs = {np.count_nonzero(np.isnan(X_train))}")
# Total number of NaNs = 79

for col in range(X_train.shape[1]):
    print(f"feature: {col + 1} has {np.count_nonzero(np.isnan(X_train[:, col]))} NAs")
'''
feature: 1 has 19 NAs
feature: 2 has 20 NAs
feature: 3 has 20 NAs
feature: 4 has 20 NAs
'''

# Initialize a new (default) LCE classifier-
lce_clf = LCEClassifier()

# Train model on missing data-
lce_clf.fit(X_train, y_train)

# Make predictions-
y_pred = lce_clf.predict(X_test)

# Compute metrics on test data-
acc = accuracy_score(y_true = y_test, y_pred = y_pred)
prec = precision_score(y_true = y_test, y_pred = y_pred, average = 'macro')
rec = recall_score(y_true = y_test, y_pred = y_pred, average = 'macro')

print("Default LCE classifier metrics on test dataset (trained on missing training data):")
print(f"accuracy = {acc * 100:.2f}%, precision = {prec * 100:.2f}% &"
f" recall = {rec * 100:.2f}%")
# Default LCE classifier metrics on test dataset (trained on missing training data):
# accuracy = 92.11%, precision = 92.75% & recall = 91.86%

print(f"Confusion matrix (trained on missing training data):\n{confusion_matrix(y_true = y_test, y_pred = y_pred)}")
'''
Confusion matrix (trained on missing training data):
[[11  0  0]
 [ 0 15  1]
 [ 0  2  9]]
'''

# Use 3-fold cross-validation using scikit model selection-
print(f"3-fold cross-validation score (trained on missing data) = {cross_val_score(lce_clf, iris_data.data, iris_data.target, cv = 3)}")
# 3-fold cross-validation score (trained on missing data) = [0.98 0.94 0.94]




# LCE regression:

# Load diabetes data-
diabetes_data = load_diabetes()

X_train, X_test, y_train, y_test = train_test_split(diabetes_data.data, diabetes_data.target)

X_train.shape, y_train.shape
# ((331, 10), (111,))

X_test.shape, y_test.shape
# ((331,), (111,))


# Initialize and train default LCE classifier-
lce_reg = LCERegressor()
lce_reg.fit(X_train, y_train)

# Make predictions-
y_pred = lce_reg.predict(X_test)

# Compute metrics on test data-
mse = mean_squared_error(y_true = y_test, y_pred = y_pred)
mae = mean_absolute_error(y_true = y_test, y_pred = y_pred)

print(f"Testing metrics: MSE = {mse:.4f} & MAE = {mae:.4f}")
# Testing metrics: MSE = 3548.6088 & MAE = 48.7135

