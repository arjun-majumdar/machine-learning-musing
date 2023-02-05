

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


"""
Clustering example and hyper-parameter tuning with K-Means and DBSCAN algorithhms.


Reference-
https://stackoverflow.com/questions/15050389/estimating-choosing-optimal-hyperparameters-for-dbscan
https://towardsdatascience.com/machine-learning-clustering-dbscan-determine-the-optimal-value-for-epsilon-eps-python-example-3100091cfbc
https://stackoverflow.com/questions/15050389/estimating-choosing-optimal-hyperparameters-for-dbscan/15063143#15063143
https://medium.com/@mohantysandip/a-step-by-step-approach-to-solve-dbscan-algorithms-by-tuning-its-hyper-parameters-93e693a91289
"""


"""
Silhouette metric is a distance calculation algorithm using euclidean or
Manhattan distance. A Silhouette Score always ranges between -1 to 1. A high
Silhouette score suggests that the objects are well matched to their own
cluster and poorly matched to their neighborhood clusters.
"""

# Create synthetic data-
X, y = make_blobs(n_samples = 100000, n_features = 10)

X.shape, y.shape
# ((100000, 10), (100000,))

for i in range(X.shape[1]):
    print(f"feature: {i}; min = {X[i].min():.4f} & max = {X.max():.4f}")
'''
feature: 0; min = -9.2403 & max = 11.8650
feature: 1; min = -7.1044 & max = 11.8650
feature: 2; min = -6.5124 & max = 11.8650
feature: 3; min = -8.1642 & max = 11.8650
feature: 4; min = -9.0917 & max = 11.8650
feature: 5; min = -7.1703 & max = 11.8650
feature: 6; min = -10.3007 & max = 11.8650
feature: 7; min = -8.3399 & max = 11.8650
feature: 8; min = -8.1128 & max = 11.8650
feature: 9; min = -8.5871 & max = 11.8650
'''

# Scale dataset using standard scaler-
std_scaler = StandardScaler()
X_std = std_scaler.fit_transform(X)


# K-Means clustering method:
# Find optimum k with elbow method and silhouette score-
wcss_l = []
silhouette_scores = []
for k in range(2, 20, 2):
    km = KMeans(n_clusters = k, init = 'k-means++')
    labels = km.fit_predict(X_std)
    wcss_l.append(km.inertia_)
    silhouette_scores.append(silhouette_score(X, labels))


# Visual elbows-
plt.plot(list(range(2, 20, 2)), wcss_l)
plt.xlabel("k")
plt.ylabel("wcss")
plt.title("Optimal k - WCSS, elbow method")
plt.show()

plt.plot(list(range(2, 20, 2)), silhouette_scores)
plt.xlabel("k")
plt.ylabel("ss")
plt.title("Optimal k - Silhouette score, elbow method")
plt.show()




# DBSCAN clustering method:

"""
In order to determine the best value of eps for your dataset, use
the K-Nearest Neighbours approach as explained in these two papers-
Sander et al. 1998 and Schubert et al. 2017 (both papers from the
original DBSCAN authors).

Here's a condensed version of their approach: If you have N-dimensional
data to begin, then choose n_neighbors in sklearn.neighbors.NearestNeighbors
to be equal to 2xN - 1, and find out distances of the K-nearest neighbors
(K being 2xN - 1) for each point in your dataset. Sort these distances out
and plot them to find the 'elbow' which separates noisy points (with high
K-nearest neighbor distance) from points (with relatively low K-nearest
neighbor distance) which will most likely fall into a cluster. The distance
at which this 'elbow' occurs is your point of optimal eps.
"""
def get_kdist_plot(X=None, k=None, radius_nbrs=1.0):

    nbrs = NearestNeighbors(n_neighbors = k, radius = radius_nbrs).fit(X)

    # For each point, compute distances to its k-nearest neighbors
    distances, indices = nbrs.kneighbors(X)

    distances = np.sort(distances, axis = 0)
    distances = distances[:, k-1]

    return distances



# k = 2*{dim(dataset)} - 1
k = 2 * X.shape[-1] - 1
distances = get_kdist_plot(X = X_std, k = k)


# Plot the sorted K-nearest neighbor distance for each point in the dataset
plt.figure(figsize = (8,8))
plt.plot(distances)
plt.xlabel('Points/Objects in the dataset', fontsize = 12)
plt.ylabel('Sorted {}-nearest neighbor distance'.format(k), fontsize = 12)
plt.grid(True, linestyle = "--", color = 'black', alpha = 0.4)
plt.title("Optimal epsilon - DBSCAN algorithm")
plt.show()


"""
NOTE: I would strongly advice the reader to refer to the two papers
cited above (especially Schubert et al. 2017) for additional tips on
how to avoid several common pitfalls when using DBSCAN as well as other
clustering algorithms.

There are a few articles online –– DBSCAN Python Example: The Optimal Value
For Epsilon (EPS) and CoronaVirus Pandemic and Google Mobility Trend EDA –– which
basically use the same approach but fail to mention the crucial choice of
the value of K or n_neighbors as 2xN-1 when performing the above procedure.


min_samples hyperparameter-
As for the min_samples hyperparameter, I agree with the suggestions in the
accepted answer. Also, a general guideline for choosing this hyperparameter's
optimal value is that it should be set to twice the number of features
(Sander et al. 1998). For instance, if each point in the dataset has 10 features,
a starting point to consider for min_samples would be 20.
"""
# Different epsilons values to explore-
eps_l = np.arange(0.6, 0.9, 0.1)

for e in eps_l:
    db_model = DBSCAN(eps = e, min_samples = 20).fit(X_std)
    core_samples_mask = np.zeros_like(db_model.labels_, dtype = bool)
    core_samples_mask[db_model.core_sample_indices_] = True
    labels = db_model.labels_
    sil_avg = silhouette_score(X, labels)
    print(f"eps = {e:.3f}, # of clusters = {len(set(labels)) - 1} & "
          f"avg silhouette score = {sil_avg:.4f}")
'''
eps = 0.600, # of clusters = 3 & avg silhouette score = 0.7106
eps = 0.700, # of clusters = 3 & avg silhouette score = 0.7513
eps = 0.800, # of clusters = 3 & avg silhouette score = 0.7572
eps = 0.900, # of clusters = 3 & avg silhouette score = 0.7652
'''
del db_model

# Initialize a new model with 'optimal' parameters-
dbs_model = DBSCAN(eps = 0.9, min_samples = 20)
labels = dbs_model.fit_predict(X_std)

# Count distribution of labels-
num, cnt = np.unique(labels, return_counts = True)

num, cnt
# (array([-1,  0,  1,  2]), array([   23, 33323, 33330, 33324]))

# Use TSNE for dimensionality reduction-
tsne_model = TSNE(
    n_components = 3, perplexity = 30.0,
    early_exaggeration = 12.0, learning_rate = 'auto',
    n_iter = 1000, n_iter_without_progress = 300,
    min_grad_norm = 1e-07, metric = 'euclidean',
    init = 'pca'
)
X_std_comp = tsne_model.fit_transform(X_std)


plt.scatter(x = X_std_comp[:, 0], y = X_std_comp[:, 1], label = labels)
plt.title("TSNE reduced dimensionality - DBSCAN clustering")
plt.show()

