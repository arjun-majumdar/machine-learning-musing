

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from scipy import linalg
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.datasets import make_blobs
from sklearn.preprocessing import normalize


"""
Manual Spectral Clustering in Python

Refer-
https://www.youtube.com/watch?v=YHz0PHcuJnk
"""


# Create dataset for clustering-
X, y = make_blobs(
    n_samples = 2000, n_features = 2,
    cluster_std = 1.2, shuffle = True
    )

X.shape, y.shape
# ((2000, 2), (2000,))

# Anisotropocally distributed data-
transformation = [
    [0.60834549, -0.63667341],
    [-0.40887718, 0.85253229]
    ]

X_anisotropic = np.dot(X, transformation)

# View data distribution as a scatter plot-
plt.scatter(X[:, 0], X[:, 1])
plt.show()


# Use K-Means clustering-
model_km = KMeans(n_clusters = 3)
pred_km = model_km.fit_predict(X_anisotropic)

# Visulize K-Means clustering result-
plt.scatter(X[:, 0], X[:, 1], c = pred_km)
plt.show()


# Use Spectral clustering-
model_spec = SpectralClustering(n_clusters = 3, assign_labels = 'discretize')
pred_spec = model_spec.fit_predict(X_anisotropic)

# Visulize Spectral clustering result-
plt.scatter(X[:, 0], X[:, 1], c = pred_spec)
plt.show()


# Spectral clustering implementation:

# Step-1: Compute Similarity Graph/Matrix-
# Given x1, ..., xn, compute a similarity graph with adjacency matrix
# K (belongs to) R^(nxn)
# hyperparameter: r > 0, decay rate -> plot(distance (x-axis) vs. similarity
# (y-axis))
r_hyperparam = 1.0

# Compute K (adjacency matrix)-
K = np.exp(
    r_hyperparam * distance.cdist(X_anisotropic, X_anisotropic, metric = 'minkowski')
    )
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html

K.shape
# (2000, 2000)

K.max(), K.min()
# (1058922360310.6765, 1.0)

# Compute diagonal matrix (non-diagonal elements are 0s)-
D = K.sum(axis = 1) # sum along rows
# Used to normalize the adjacency matrix

D.shape
# (2000,)

D = np.sqrt(1 / D)

D.max(), D.min()
# (4.835020147002329e-05, 1.554269073723542e-07)

# Efficient implementation by converting 'D' from 1D to 2D-
M = np.multiply(D[np.newaxis, :], np.multiply(K, D[:, np.newaxis]))
# Normalized, adjacency matrix

M.shape
# (2000, 2000)

# D[np.newaxis:, ].shape
# (2000,)


"""
# Fast matrix multiplication, example-
A = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
    ])

B = np.array([1, 2, 3])

# Convert 'B' to 2D matrix-
B[:, np.newaxis]
'''
array([[1],
       [2],
       [3]])
'''

# Multiply first row of A with B[0], second row with B[1] & third row
# with B[2]-
np.multiply(A, B[:, np.newaxis])
"""


"""
Relation to normalized Laplacian matrix-

L = In - D^(-1/2)*K*D^(-1/2) = In - M
M = In - L

In = Identity matrix, M = normalized matrix from above 
"""


# Step-2: Eigen-value decomposition (matrix factorization)-
# Compute EVD of M-
U, Sigma, _ = linalg.svd(M, full_matrices = False, lapack_driver = 'gesvd')
"""
Orthogonal matrix 'U': u1, u2, ... uk -> eigen vectors (n rows, k columns)

Diagonal matrix 'Sigmal'-
M = UΣUT -> a matrix where diagonal elements are: σ1, ... σn, and
non-diagonal elements are 0s.

The singular values are sorted in decreasing order.
When you use 'M' matrix, you are going to keep eigen-vectors corresponding
to the largest eigen-values of M matrix. It is common to keep exactly the
same number of eigen-vectors as the number of clusters you want to have.
"""

U.shape, Sigma.shape
# ((2000, 2000), (2000,))

# Since this dataset has 3 cluters, we do-
U_subset = U[:, 0:3]

# U_subset.shape
# (2000, 3)


# Step-3: Do K-Means clustering-
model_km = KMeans(n_clusters = 3)
pred_spec_manual = model_km.fit_predict(normalize(U_subset))
# normalize each row to 1.0

# Visualize results-
plt.scatter(X[:, 0], X[:, 1], c = pred_spec_manual, s = 20)
plt.title("Manual Spectral Clustering output")
plt.show()


