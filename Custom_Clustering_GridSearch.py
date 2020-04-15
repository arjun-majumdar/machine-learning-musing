

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_blobs
from sklearn.pipeline import Pipeline

import numpy as np
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV




class KMeansTransformer(BaseEstimator, TransformerMixin):

	def __init__(self, **kwargs):
		# The purpose of 'self.model' is to contain the
		# underlying cluster model-
		self.model = KMeans(**kwargs)
		

	def fit(self, X):
		self.X = X
		self.model.fit(X)


	def transform(self, X):
		pred = self.model.predict(X)
		return np.hstack([self.X, pred.reshape(-1, 1)])


	def fit_transform(self, X, y=None):
		self.fit(X)
		return self.transform(X)




# Create features and target-
X, y = make_blobs(n_samples=100, n_features=2, centers=3)

# Get shape/dimension-
X.shape, y.shape
# ((100, 2), (100,))


# Create another pipeline using Decision Tree as classifier-
pipe_dt = Pipeline(
	[
		('sc', StandardScaler()),
		('kmt', KMeansTransformer()),
		('dt_clf', DecisionTreeClassifier())
	]
)

# Train defined pipline-
pipe_dt.fit(X, y)

# Get accuracy score of pipeline-
pipe_dt.score(X, y)
# 1.0

# Make predictions using pipeline defined above-
y_pred_dt = pipe_dt.predict(X)


'''
# Visualize predictions using Decision Tree classifier pipeline-
plt.scatter(X[:,0], X[:,1], c = y_pred_dt)

plt.title("Dataset Visualization - DecisionTree pipeline")
plt.xlabel("x-coordinates")
plt.ylabel("y-coordinates")
plt.grid()
plt.show()
'''




# Perform hyperparameter search/optimization using 'GridSearchCV'-

# Specify parameters to be hyper-tuned-
params = {
			'n_clusters': [2, 3, 5, 7]
			}

# Initialize GridSearchCV() object using 3-fold CV-
grid_kmt = GridSearchCV(param_grid=params, estimator=pipe_dt, cv = 3)

# Perform GridSearchCV on training data-
grid_kmt.fit(X, y)

