

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
# from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


# Read in data-
iris_data = pd.read_csv("iris.csv")
# OR-
# Assign colum names to the dataset
# colnames = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
# iris_data = pd.read_csv("iris.csv", names = colnames)

# To get unique values for an attribute, example, say for 'class'-
# iris_data['class'].unique()

# To get distribution of unique values for attribute 'class'-
# iris_data['class'].value_counts()

iris_data.shape
# (150, 5)

iris_data.dtypes
'''
sepallength    float64
sepalwidth     float64
petallength    float64
petalwidth     float64
class           object
dtype: object
'''

# Check missing values-
iris_data.isna().values.any()
# False

iris_data.isna().sum().sum()
# 0

# Get distribution of target attribute-
iris_data['class'].value_counts()
'''
Iris-setosa        50
Iris-versicolor    50
Iris-virginica     50
Name: class, dtype: int64
'''


"""
# Visualize distribution of data-
# To visualize a categorical attribute 'class' in data-
iris_data['class'].value_counts().plot(kind = 'bar')
# OR-
# iris_data.groupby('class').size().plot(kind = 'bar')

# OR, using 'seaborn'-
# sns.countplot(iris_data['class'])
# sns.countplot(iris_data['class'], color = 'gray')
# OR-
# 'seaborn' also supports coloring the bars in the right color with a little trick-
# sns.countplot(iris_data['class'], palette={color: color for color in iris_data['class'].unique()})


# Show/draw the graph-
plt.show()
"""

# Select numberic attributes-
iris_data.select_dtypes(include = np.number).columns
# Index(['sepallength', 'sepalwidth', 'petallength', 'petalwidth'], dtype='object')

# Select non-numberic attributes-
iris_data.select_dtypes(include = object).columns
# Index(['class'], dtype='object')

# Get basic statistics for numeric columns in dataset-
iris_data.describe()
'''
       sepallength  sepalwidth  petallength  petalwidth
count   150.000000  150.000000   150.000000  150.000000
mean      5.843333    3.054000     3.758667    1.198667
std       0.828066    0.433594     1.764420    0.763161
min       4.300000    2.000000     1.000000    0.100000
25%       5.100000    2.800000     1.600000    0.300000
50%       5.800000    3.000000     4.350000    1.300000
75%       6.400000    3.300000     5.100000    1.800000
max       7.900000    4.400000     6.900000    2.500000
'''

# Visualize numeric columns using boxplots-
sns.boxplot(data = iris_data)
plt.title("Iris dataset: boxplots")
plt.show()
"""
'sepallength' has an almost gaussian distribution
'sepalwidth' also has an almost gaussian distribution but with outliers
'petallength' & 'petalwidth' are left-skewed distributions
"""

# Visualize distributions for each numeric columns-
num_vals, bins, patches = plt.hist(iris_data['sepallength'], bins = int(np.ceil(np.sqrt(len(iris_data)))))
plt.xlabel("sepal length")
plt.ylabel("count")
plt.title("Distribution: sepal length")
plt.show()

num_vals, bins, patches = plt.hist(iris_data['sepalwidth'], bins = int(np.ceil(np.sqrt(len(iris_data)))))
plt.xlabel("sepal width")
plt.ylabel("count")
plt.title("Distribution: sepal width")
plt.show()

num_vals, bins, patches = plt.hist(iris_data['petallength'], bins = int(np.ceil(np.sqrt(len(iris_data)))))
plt.xlabel("petal length")
plt.ylabel("count")
plt.title("Distribution: petal length")
plt.show()

num_vals, bins, patches = plt.hist(iris_data['petalwidth'], bins = int(np.ceil(np.sqrt(len(iris_data)))))
plt.xlabel("petal width")
plt.ylabel("count")
plt.title("Distribution: petal width")
plt.show()

# According to distribution visualizations from above, appropriate scalers are used-
std_scaler = StandardScaler()
iris_data[['sepallength', 'sepalwidth']] = std_scaler.fit_transform(iris_data[['sepallength', 'sepalwidth']])

# 'StandardScaler' subtracts the mean from each feature/attribute and then
# scales to unit variance

# Sanity checks-
iris_data['sepallength'].min(), iris_data['sepallength'].max()
# (-1.870024133847019, 2.4920192021244283)

iris_data['sepalwidth'].min(), iris_data['sepalwidth'].max()
# (-2.438987252491841, 3.1146839106774356)


mm_scaler = MinMaxScaler()
iris_data[['petallength', 'petalwidth']] = mm_scaler.fit_transform(iris_data[['petallength', 'petalwidth']])

# Sanity checks-
iris_data['petallength'].min(), iris_data['petallength'].max()
# (0.0, 1.0)

iris_data['petalwidth'].min(), iris_data['petalwidth'].max()
# (0.0, 1.0)


# Label encode target attribute-
le = LabelEncoder()
iris_data['class'] = le.fit_transform(iris_data['class'])

# le.inverse_transform(iris_data['class'])


# Divide the data into attributes (X) and labels (y)-
X = iris_data.drop('class', axis = 1)
y = iris_data['class']


# Divide attributes & labels into training & testing sets-
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size = 0.3, stratify = y
    )

# Sanity checks-
X_train.shape, y_train.shape
# ((105, 4), (105,))

X_test.shape, y_test.shape
# ((45, 4), (45,))


# Use k-NN model for classification-

"""
# Instantiate a kNN classifier using 3 nearest neighbors-
knn_cls = KNeighborsClassifier(n_neighbors=3)

# Train/fit model to training data-
knn_cls.fit(X_train, y_train)

# Make predictions using trained model-
y_pred = knn_cls.predict(X_test)


# Print accuracy, confusion matrix of trained model-
print("\n\nAccuracy achieved by model = {0:.4f}%\n\n".format(accuracy_score(y_test, y_pred) * 100))
print("\nConfusion matrix =\n", confusion_matrix(y_test, y_pred))
"""




# Performing Randomized Search for hyper parameter tuning-
search_parameters = {
    'n_neighbors': [x for x in range(2, 41, 2)],
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    'metric': ['minkowski', 'euclidean', 'manhattan']
    }


print("\n\nPerforming RandomizedSearch now to find best parameters.\n")

# RandomizedSearchCV-
rs_knn = RandomizedSearchCV(
    KNeighborsClassifier(), param_distributions = search_parameters,
    cv = 5, n_iter = 100
    )

"""
The most important arguments in 'RandomizedSearchCV()' are-
1.) 'n_iter', which controls the number of different combinations to try
2.) 'cv' which is the number of folds to use for cross validation

More iterations 'n_iter' will cover a wider search space and more 'cv' folds reduces the
chances of overfitting, but raising each will increase the run time.

Machine learning is a field of trade-offs, and performance vs time is one of the most fundamental.
"""

# Fit the random search model-
rs_knn.fit(X_train, y_train)

# We can view the best parameters from fitting the random search 'rs_gbm.best_params_'-
print("\n\nBest Parameters found using RandomizedSearch are: \n{0}\n\n".format(rs_knn.best_params_))
"""
Best Parameters found using RandomizedSearch are: 
{'weights': 'distance', 'n_neighbors': 10, 'metric': 'manhattan', 'algorithm': 'brute'}
"""


# Using parameters found above to perform Grid Search-
best_parameters = {
    'weights': ['distance'],
    'n_neighbors': [8, 9, 10, 11, 12],
    'metric': ['manhattan'],
    'algorithm': ['brute']
    }


# Cross-Validated Grid Search using 10-fold cross-validation-
cv_knn = GridSearchCV(
    estimator = KNeighborsClassifier(), param_grid = best_parameters,
    cv = 10
    )

# Train/Fit model on training data-
cv_knn.fit(X_train, y_train)

# Get best parameters found using Grid-Search technique-
best_params = cv_knn.best_params_
print("\n\nBest Parameters found using GridSearch:\n{0}\n\n".format(best_params))
"""
Best Parameters found using GridSearch:
{'algorithm': 'brute', 'metric': 'manhattan', 'n_neighbors': 10, 'weights': 'distance'}
"""


# Finally, train best knn classifier using hyper parameters found using Grid Search-
best_knn_classifier = KNeighborsClassifier(
    algorithm = cv_knn.best_params_['algorithm'], metric = cv_knn.best_params_['metric'],
    n_neighbors = cv_knn.best_params_['n_neighbors'], weights = cv_knn.best_params_['weights']
    )

# Train best knn to training data-
best_knn_classifier.fit(X_train, y_train)

# Make predictions using trained model-
y_pred_best = best_knn_classifier.predict(X_test)

# Compute performance metrics of 'best' trained model-
acc = accuracy_score(y_true = y_test, y_pred = y_pred_best)
prec = precision_score(y_true = y_test, y_pred = y_pred_best, average = 'macro')
rec = recall_score(y_true = y_test, y_pred = y_pred_best, average = 'macro')

print("kNN 'best' model's performance on validation data:")
print(f"accuracy = {acc * 100:.2f}%, precision = {prec * 100:.2f}% & recall = {rec * 100:.2f}%")
# kNN 'best' model's performance on validation data:
# accuracy = 88.89%, precision = 88.99% & recall = 88.89%

print(f"Confusion Matrix (best model):\n{confusion_matrix(y_true = y_test, y_pred = y_pred_best)}")
'''
Confusion Matrix (best model):
[[15  0  0]
 [ 0 12  3]
 [ 0  2 13]]
'''




# Brute for kNN for best value of 'k'-

best_score = {}

for k in range(2, 41, 2):
	knn_cls = KNeighborsClassifier(n_neighbors = k)
	knn_cls.fit(X_train, y_train)
	y_pred = knn_cls.predict(X_test)
	best_score[k] = accuracy_score(y_test, y_pred)


# Plot k-values Versus accuracy scores-
plt.plot(best_score.keys(), best_score.values())
plt.xlabel("K-Values")
plt.ylabel("Accuracy Scores")
plt.title("kNN: brute-forcing 'k'")
plt.show()
# Plot shows that one of best values for k = 2


# Training a knn using k=2-
knn = KNeighborsClassifier(
    n_neighbors = 2, weights = 'uniform',
    algorithm = 'auto', leaf_size = 30,
    p = 2, metric = 'minkowski'
    )
knn.fit(X_train, y_train)

# Make predictions using trained model-
y_pred_best = knn.predict(X_test)

# Compute performance metrics of 'best' trained model-
acc = accuracy_score(y_true = y_test, y_pred = y_pred_best)
prec = precision_score(y_true = y_test, y_pred = y_pred_best, average = 'macro')
rec = recall_score(y_true = y_test, y_pred = y_pred_best, average = 'macro')

print("kNN 'best' model's performance on validation data:")
print(f"accuracy = {acc * 100:.2f}%, precision = {prec * 100:.2f}% & recall = {rec * 100:.2f}%")
# kNN 'best' model's performance on validation data:
# accuracy = 91.11%, precision = 91.55% & recall = 91.11%

print(f"Confusion Matrix (best model):\n{confusion_matrix(y_true = y_test, y_pred = y_pred_best)}")
'''
Confusion Matrix (best model):
[[15  0  0]
 [ 0 14  1]
 [ 0  3 12]]
'''

