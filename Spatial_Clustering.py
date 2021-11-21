

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors
import scipy.cluster.hierarchy as sc
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
# import gower as gower


'''
DBSCAN hyper-parameter optimization.


Dataset-
https://github.com/dbvis-ukon/movekit/edit/master/examples/datasets/fish-5-cleaned.csv


Refer-
https://towardsdatascience.com/machine-learning-clustering-dbscan-determine-the-optimal-value-for-epsilon-eps-python-example-3100091cfbc
https://www.kaggle.com/questions-and-answers/166388
'''


data = pd.read_csv("fish_5_cleaned.csv")

data.shape
# (5000, 4)

data.dtypes
'''
time           int64
animal_id      int64
x            float64
y            float64
dtype: object
'''

data['time'].min(), data['time'].max()
# (1, 1000)

data['animal_id'].unique()
# array([312, 511, 607, 811, 905], dtype=int64)

data['x'].min(), data['x'].max()
# (52.91, 762.44)

data['y'].min(), data['y'].max()
# (33.74, 485.86)

# columns/features to be used for clustering-
cols = ['animal_id', 'x', 'y']

data = data.loc[:, cols]

data.shape
# (5000, 3)


'''
# OR-
data = pd.read_csv("fish_5_cleaned.csv", usecols = ['animal_id', 'x', 'y'])

data.shape
# (5000, 3)

data.columns
# Index(['animal_id', 'x', 'y'], dtype='object')
'''


# Visualize 'x' & 'y' coordinates-
# sns.scatterplot(data = data, x = 'x', y = 'y', hue = 'animal_id')

plt.scatter(x = data['x'], y = data['y'], c = data['animal_id'])
# OR-
# colors = {312: 'green', 511: 'red', 607: 'yellow', 811: 'blue', 905: 'black'}
# plt.scatter(x = data['x'], y = data['y'], c = data['animal_id'].map(colors))

plt.xlabel("x coordinates")
plt.ylabel("y coordinates")
plt.title("fish dataset: x & y coordinates")
plt.show()


# Visualize distribution of 'x' column-
plt.boxplot(x = data['x'])
plt.show()

plt.hist(data['x'], bins = int(np.ceil(np.sqrt(len(data)))))
plt.show()

# Visualize distribution of 'y' column-
plt.hist(data['y'], bins = int(np.ceil(np.sqrt(len(data)))))
plt.show()


# Visualize the 3 columns/features as a 3-D scatter plot-  
fig = px.scatter_3d(
    data, x = 'x',
    y = 'y', z = 'animal_id',
    color = 'animal_id'
    )  
fig.show()


# Scale columns-
rb_scaler = RobustScaler()
data_scaled = rb_scaler.fit_transform(data)
# data_scaled = rb_scaler.fit_transform(data.loc[:, ['x', 'y']])

data_scaled.shape
# (5000, 3)

# Convert from numpy array to pandas DF-
data_scaled = pd.DataFrame(data_scaled, columns = ['animal_id', 'x', 'y'])

'''
# Concate column-wise to get the final numpy array-
data_scaled = np.concatenate((data_scaled, data['animal_id'].values.reshape(5000, 1)), axis = 1)

data_scaled.shape
# (5000, 3)
'''

# Sanity check that scaling worked-
data_scaled.describe()
'''
         animal_id            x             y
count  5000.000000  5000.000000  5.000000e+03
mean      0.074000    -0.157276 -2.697733e-01
std       0.705914     0.648809  5.642866e-01
min      -0.983333    -1.082606 -1.422012e+00
25%      -0.320000    -0.842456 -8.237342e-01
50%       0.000000     0.000000  1.170803e-16
75%       0.680000     0.157544  1.762658e-01
max       0.993333     1.367958  4.406336e-01
'''




# Agglomerative Clustering

# Visualize dendrogram-
dendrogrm = sc.dendrogram(sc.linkage(data, method = 'single'))
plt.title('Dendrogram')
plt.xlabel("data points")
plt.ylabel("distance")
plt.title("Dendrogram: single linkage")
plt.show()

# The dendrogram suggests that the 'optimal' number of clusters for this dataset is 5.

# method = 'single', 'complete', 'average', 'weighted', 'centroid', 'median', 'ward'


# Train a model using Agglomerative Hierarchical Clustering-

# Without specifying number of clusters-
model = AgglomerativeClustering(affinity = 'euclidean', linkage = 'single')
pred = model.fit_predict(data)
# NOTE: 'n_clusters = 2' (default value)

set(pred)
# {0, 1}

np.unique(pred, return_counts = True)
# (array([0, 1], dtype=int64), array([2717, 2283], dtype=int64))


# Specify number of clusters according to visual inspection of dendrogram-

# Initialize an AgglomerativeClustering object-
model = AgglomerativeClustering(
    n_clusters = 5, affinity = 'euclidean',
    linkage = 'single'
    )

# Train and predict on training data-
pred = model.fit_predict(data)

set(pred)
# {0, 1, 2, 3, 4}

np.unique(pred, return_counts = True)
'''
(array([0, 1, 2, 3, 4], dtype=int64),
 array([1000, 1000, 1000, 1000, 1000], dtype=int64))
'''

# Visualize clustering outcome-
plt.scatter(x = data['x'], y = data['y'], c = pred)
plt.xlabel("x coordinate")
plt.ylabel("y coordinate")
plt.title("Agglomerative Clustering: single linkage")
plt.show()

# Conclusion: for this particular dataset, 'single linkage' distance measure as the inter-cluster measure
# seems to be the optimal option. Agglomerative clustering using this distance measure seems to be
# outperforming other clustering algorithms such as KMeans & DBSCAN.
# KMeans algorithm is not appropriate for this dataset due to the dataset.
# Probably further hyper-parameter tuning for DBSCAN might yield the correct clustering output.




# Use K-Means hyper-parameter tuning for 'k'-

# Use inertia to find optimal 'k'-
inertia_d = {}

for k in list(range(2, 11, 2)):
    km = KMeans(n_clusters = k, init = 'k-means++')
    km.fit(data_scaled)
    inertia_d[k] = km.inertia_
    print(f"\nk = {k}; cluster distribution: {np.unique(km.labels_, return_counts = True)}")
'''
k = 2; cluster distribution: (array([0, 1]), array([3613, 1387], dtype=int64))

k = 4; cluster distribution: (array([0, 1, 2, 3]), array([2194,  588,  806, 1412], dtype=int64))

k = 6; cluster distribution: (array([0, 1, 2, 3, 4, 5]), array([ 713, 1099, 1113,  558,  788,  729], dtype=int64))

k = 8; cluster distribution: (array([0, 1, 2, 3, 4, 5, 6, 7]), array([ 395,  713,  558,  729,  476, 1059,  703,  367], dtype=int64))

k = 10; cluster distribution: (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), array([670, 720, 448, 358, 690, 710, 366, 420, 262, 356], dtype=int64))
'''

# Visualize elbow plot using inertia-
plt.plot(inertia_d.keys(), inertia_d.values())
plt.xlabel("k")
plt.ylabel("inertia")
plt.title("K-Means - Elbow plot with inertias")
plt.show()

# optimal k = 4 & 6.

# Sanity check-
km = KMeans(n_clusters = 6, init = 'k-means++')
labels = km.fit_predict(data_scaled)

data_scaled['labels'] = labels

# Visualize the 3 columns/features as a 3-D scatter plot using KMeans clustered labels-  
fig = px.scatter_3d(
    data_scaled, x = 'x',
    y = 'y', z = 'animal_id',
    color = 'labels'
    )  
fig.show()

# With k = 4 & 6, the resulting 3-D scatter plot when compared with original data shows that K-Means
# is not the most optimal clustering algo for this particular dataset.




# Use DMDBSCAN algo to get 'optimal' values of 'eps' & 'min_samples' for
# DBSCAN algo.
'''
We can calculate the distance from each point to its closest neighbour using
the 'NearestNeighbors'. The point itself is included in 'n_neighbors'. The
'kneighbors' method returns two arrays, one which contains the distance to
the closest 'n_neighbors' points and the other which contains the index for
each of those points.
'''
neigh = NearestNeighbors(n_neighbors = 6)
# neigh = NearestNeighbors(n_neighbors = 7)
# nbrs = neigh.fit(data_scaled)
nbrs = neigh.fit(data)
# distances, indices = nbrs.kneighbors(data_scaled)
distances, indices = nbrs.kneighbors(data)

# Next, we sort and plot results-
distances = np.sort(distances, axis = 0)
distances = distances[:, 1]

# Visualize eps vs. distance as an elbow plot-
plt.plot(distances)
plt.xlabel("distances")
plt.ylabel("eps")
plt.title("eps - Visualization")
plt.show()


# Gather a set of eps for hyper-parameter tuning-
'''
epsilons = [2e-06, 3.5e-05, 4.3e-05, 5.4e-05, 7.0e-05, 8.2e-05, 8.4e-05, 9.1e-05, 0.000105, 0.000109, 0.000113,
0.000139, 0.000162, 0.000179, 0.000256, 0.000321
]

epsilons = [0.00252, 0.00227, 0.00246, 0.00284, 0.00312, 0.00322, 0.00353, 0.00379
]
'''
epsilons = [
    0.24, 0.48, 0.483
]

for eps in epsilons:
    dbm = DBSCAN(eps = eps, min_samples = 7)
    # labels = dbm.fit_predict(data_scaled)
    labels = dbm.fit_predict(data)
    print(f"\neps = {eps} has {len(set(labels)) - 1} clusters")
    print(f"cluster distribution: {np.unique(dbm.labels_, return_counts = True)}")
'''
eps = 2e-05 has 0 clusters
cluster distribution: (array([-1], dtype=int64), array([5000], dtype=int64))

eps = 5e-05 has 1 clusters
cluster distribution: (array([-1,  0], dtype=int64), array([4993,    7], dtype=int64))

eps = 8e-05 has 3 clusters
cluster distribution: (array([-1,  0,  1,  2], dtype=int64), array([4974,    9,    8,    9], dtype=int64))

eps = 0.00015 has 6 clusters
cluster distribution: (array([-1,  0,  1,  2,  3,  4,  5], dtype=int64), array([4930,    7,   17,   14,   11,   11,   10], dtype=int64))

eps = 0.00018 has 7 clusters
cluster distribution: (array([-1,  0,  1,  2,  3,  4,  5,  6], dtype=int64), array([4911,    9,   21,   17,    8,   13,   11,   10], dtype=int64))
'''

# Sanity check-
dbm = DBSCAN(eps = 0.00015, min_samples = 7)
labels = dbm.fit_predict(data_scaled)

data_scaled['labels'] = labels

