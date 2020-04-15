

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_blobs
from sklearn.pipeline import Pipeline

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from sklearn.metrics import accuracy_score, precision_score, recall_score
import time

from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


'''

Refer-
https://stellasia.github.io/blog/2020-02-02-wrapping-custom-mode-into-sklearn-estimator/
https://leoliu1221.wordpress.com/2016/12/13/how-to-make-custom-estimator-class-and-custom-scorer-to-do-cross-validation-using-sklearn-api-on-your-custom-model/
http://danielhnyk.cz/creating-your-own-estimator-scikit-learn/
https://towardsdatascience.com/building-a-custom-model-in-scikit-learn-b0da965a1299

G: creating custom estimator sklearn
'''




class ST_DBSCAN(BaseEstimator, TransformerMixin):
# class ST_DBSCAN(BaseEstimator, ClassifierMixin):
	"""
	Spatio-Temporal DBSCAN algorithm for scikit-learn compatibility
	# eps1 spatial neighborhood
	# eps2 Time Neighborhood
	# minPts the minimum number of points satisfying the double neighborhood


	All estimators must have 'get_params()'' and 'set_params()' functions. They are inherited
	when you subclass 'BaseEstimator' and the recommendation is not to override these function
	(just not state them in definition of your classifier).
	"""

	def __init__(self, eps1 = 0.5, eps2 = 10, minPts = 5):
		'''
		All arguments must have default values, so it is possible to
		initialize the clustering object without any parameters.
		Do not take data as argument here! It should be in fit method.
		Parameters should have the same name as attributes.
		'''
		self.eps1 = eps1
		self.eps2 = eps2
		self.minPts = minPts
		self.predicted_labels = None


	def compute_squared_EDM(self, X):
		# Calculate the distance matrix of the X matrix-
		return squareform(pdist(X, metric='euclidean'))

   

	def fit(self, X, y):
		'''
		Here, you should implement all the hard work. At first you should check the parameters.
		Secondly, you should take and process the data. You'll almost surely want to add some new
		attributes to your object which are created in fit() method. These should be ended by _ at the 
		end, e.g. self.fitted_.

		And finally you should return 'self'. This is again for compatibility reasons with common interface of scikit-learn.
		'''


		# Get the rows and columns of data (a total of 'n' data)
		# n, m = self.data.shape
		self.n_, self.m_ = X.shape

		# Calculate time distance matrix
		self.timeDisMat_ = self.compute_squared_EDM(X[:,0].reshape(self.n_, 1))

		# Get space distance matrix-
		self.disMat_ = self.compute_squared_EDM(X[:, 1:])
    
		# Assign the number less than minPts in the matrix to 1, the number greater than minPts to zero, then 1 to sum each
		# row, and then find the index of the core point coordinates
		# Note: Two uses of np.where((search, replace function))
		self.core_points_index_ = np.where(np.sum(np.where((self.disMat_ <= self.eps1) & 
			(self.timeDisMat_ <= self.eps2), 1, 0), axis=1) >= self.minPts)[0]

		# Initialization category, -1 means unclassified-
		self.labels_ = np.full((self.n_,), -1)
		self.clusterId_ = 0

		# Iterate through all the core points-
		for pointId in self.core_points_index_:
			# If the core point is not classified, use it as the seed point and start to find the corresponding cluster
			if (self.labels_[pointId] == -1):
				# Mark pointId as the current category (that is, identified as operated)
				self.labels_[pointId] = self.clusterId_
            
				# Find the eps neighborhood of the seed point and the points that are not classified, and put it into the seed set-
				self.neighbour_ = np.where((self.disMat_[:, pointId] <= self.eps1) & 
					(self.timeDisMat_[:, pointId] <= self.eps2) & (self.labels_ == -1))[0]
				self.seeds_ = set(self.neighbour_)
            
				# Through seed points, start to grow, find data points with reachable density, until the seed set is empty,
				# one cluster set is searched
				while len(self.seeds_) > 0:
					# Pop up a new seed point-
					newPoint = self.seeds_.pop()

					# Mark newPoint as the current class-
					self.labels_[newPoint] = self.clusterId_

					# Find newPoint seed point eps neighborhood (including itself)
					self.queryResults_ = set(np.where((self.disMat_[:,newPoint] <= self.eps1) & 
						(self.timeDisMat_[:, newPoint] <= self.eps2) )[0])
                
					# If newPoint belongs to the core point, then newPoint can be expanded, that is, the density can be reached
					# through newPoint
					if len(self.queryResults_) >= self.minPts:
						# Push the points in the neighborhood that are not classified into the seed set
						for resultPoint in self.queryResults_:
							if self.labels_[resultPoint] == -1:
								self.seeds_.add(resultPoint)

				# After the cluster grows, find a category
				self.clusterId_ = self.clusterId_ + 1

		self.predicted_labels = self.labels_

		# return self.labels_
		return self

	
	def score(self, X, y):
		return accuracy_score(y, self.predicted_labels)

	
	def get_labels(self):
		return self.predicted_labels


	def transform(self, X):
		# pred = self.model.predict(X)
		# return np.hstack([self.X, pred.reshape(-1, 1)])
		print("\nX.shape = {0}, self.predicted_labels.shape = {1} & self.predicted_labels.reshape(-1, 1) = {2}\n".format(
			X.shape, self.predicted_labels.shape,
			self.predicted_labels.reshape(-1, 1).shape
			))
		
		return np.hstack([X, self.predicted_labels.reshape(-1, 1)])
 	



# Read in CSV file-
data = pd.read_csv("/home/arjun/University_of_Konstanz/Hiwi/Unsupervised_Learning_Works/Spatio-temporal-Clustering-master/Clustering_Ground_Truth_Data.csv")

# Take the first 1000 records-
# data_mod = data.loc[:1000, ['x', 'y']]
data_mod = data.loc[:1000, ['frame', 'x', 'y']]

data_mod.shape
#  (1001, 3)

# Get numpy values instead of Pandas DataFrame-
data_mod = data_mod.values

X = data.loc[:1000, ['frame', 'x', 'y']]
y = data.loc[:1000, 'cid']

X = X.values
y = y.values

# Get shapes-
X.shape, y.shape
# ((1001, 3), (1001,))


# Visualize dataset using only x and y coordinates with
# ground truth-
plt.scatter(X[:, 1], X[:, 2], c = y)
plt.show()


# Initialize an instance of 'ST_DBSCAN' class-
stdb = ST_DBSCAN(0.1, 60, 5)

# Perform ST-DBSCAN clustering and return labels for each
# data point-
stdb.fit(X, y)

# labels = stdb.fit(X, y)
labels = stdb.get_labels()

labels.shape
# (1001,)


X_transformed = stdb.transform(X)
# X.shape = (1001, 3), self.predicted_labels.shape = (1001,) & self.predicted_labels.reshape(-1, 1) = (1001, 1)

X_transformed.shape
# (1001, 4)


stdb.get_params().keys()
# dict_keys(['eps1', 'eps2', 'minPts'])


# Visualize dataset using only x and y coordinates with
# ST-DBSCAN predictions-
plt.scatter(X[:, 1], X[:, 2], c = X_transformed[:, 3])
plt.show()




# Create another pipeline using Decision Tree as classifier-
pipe_dt = Pipeline(
	[
		# ('sc', StandardScaler()),
		# ('kmt', KMeansTransformer()),
		('st_db', ST_DBSCAN()),
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

# Get shapes-
y_pred_dt.shape, y.shape
# ((1001,), (1001,))




# Specify parameters to be hyper-tuned-

# eps1 spatial neighborhood
# eps2 Time Neighborhood
# minPts the minimum number of points satisfying the double neighborhood

params = {
			'st_db__eps1': [0.5, 1, 1.5, 2],
			'st_db__eps2': [8, 10, 15],
			'st_db__minPts':  [5, 8, 10]
			}

# Initialize GridSearchCV() object using 3-fold CV-
# grid_kmt = GridSearchCV(param_grid=params, estimator=pipe_dt, cv = 3)
grid_kmt = GridSearchCV(param_grid=params, estimator=pipe_dt)

# Perform GridSearchCV on training data-
grid_kmt.fit(X, y)

