

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier


"""
Difference Between NearestNeighbors and KNN Classifier?

'NearestNeighbors' implements unsupervised nearest neighbors learning. It acts as a uniform
interface to three different nearest neighbors algorithms-
BallTree , KDTree , and a brute-force algorithm based on routines in sklearn's module-
'sklearn. metrics.pairwise'.

'KNN' Classifier is a type of instance-based learning or non-generalizing learning: it does not
attempt to construct a general internal model, but simply stores instances of the training data.
Classification is computed from a simple majority vote of the nearest neighbors of each point: a
query point is assigned the data class which has the most representatives within the nearest
neighbors of the point.

scikit-learn implements two different nearest neighbors classifiers- 'KNeighborsClassifier'
implements learning based on the nearest neighbors of each query point, where an integer value
is specified by the user. 'RadiusNeighborsClassifier' implements learning based on the number of
neighbors within a fixed radius of each training point, where is a floating-point value specified
by the user.


'NearestNeighbors' is an unsupervised technique of finding the nearest data points with respect
to each data point, we only fit/train the data/X in here.

'KNN Classifier' is a supervised technique of classifying a point based on distance measure to it's
'k' nearest neighbors.


Reference-
https://stackoverflow.com/questions/63134886/difference-between-nearestneighbors-and-knn-classifier
https://scikit-learn.org/stable/modules/neighbors.html
"""


# Sample data set-
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])

X.shape
# (6, 2)

X
'''
array([[-1, -1],
       [-2, -1],
       [-3, -2],
       [ 1,  1],
       [ 2,  1],
       [ 3,  2]])
'''


# Visualize dataset-
plt.scatter(x = X[:, 0], y = X[:, 1])
plt.show()


# Initialize an instance using 3 nearest neighbors-
nbrs = NearestNeighbors(n_neighbors = 3, algorithm = 'ball_tree').fit(X)

# Find distances and indices with respect to to each point-
distances, indices = nbrs.kneighbors(X)

distances.shape, indices.shape
# ((6, 3), (6, 3))


# The distances amongst each points-
distances
'''
array([[0.        , 1.        , 2.23606798],
       [0.        , 1.        , 1.41421356],
       [0.        , 1.41421356, 2.23606798],
       [0.        , 1.        , 2.23606798],
       [0.        , 1.        , 1.41421356],
       [0.        , 1.41421356, 2.23606798]])
'''


# We can see that point0 is closest to point1, and point1 is closest to point0, point2 is
# closest to point1, and so on.
# 'indices' gives us the same result. We have 3 columns as we chose neighbors = 3.
indices
'''
array([[0, 1, 2],
       [1, 0, 2],
       [2, 1, 0],
       [3, 4, 5],
       [4, 3, 5],
       [5, 4, 3]], dtype=int64)
'''


# Compute the (weighted) graph of k-Neighbors for points in X-
nbrs.kneighbors_graph(X).toarray()
'''
array([[1., 1., 1., 0., 0., 0.],
       [1., 1., 1., 0., 0., 0.],
       [1., 1., 1., 0., 0., 0.],
       [0., 0., 0., 1., 1., 1.],
       [0., 0., 0., 1., 1., 1.],
       [0., 0., 0., 1., 1., 1.]])
'''


# KNN Classifier:

X = np.array([[0], [1], [2], [3]])
y = np.array([0, 0, 1, 1])

X.shape, y.shape
# ((4, 1), (4,))

X
'''
array([[0],
       [1],
       [2],
       [3]])
'''

y
# array([0, 0, 1, 1])


# Visualize features (X) and target (y)-
plt.scatter(x = X, y = y)
plt.show()


# Initialize and train KNN classifier-
knn_clf = KNeighborsClassifier(n_neighbors = 3)
knn_clf.fit(X, y)

# Predict for point 1.1 (which cluster it belongs to)-
knn_clf.predict([[1.1]])
# array([0])

