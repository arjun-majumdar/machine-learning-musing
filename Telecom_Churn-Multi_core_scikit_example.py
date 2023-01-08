

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


"""
Example of performance benefits using multi-core ML with scikit-learn and Python.
This is a toy example and therefore does not involve proper data pre-processing,
feature engineering, etc.

Reference-
https://machinelearningmastery.com/multi-core-machine-learning-in-python/
"""


data = pd.read_csv("cell2celltrain.csv")

data.shape
# (51047, 58)

data.isna().values.any()
# True

for col in data.columns.tolist():
    if data[col].isna().values.any():
        print(f"{col} has {data[col].isna().values.sum()}")
'''
MonthlyRevenue has 156
MonthlyMinutes has 156
TotalRecurringCharge has 156
DirectorAssistedCalls has 156
OverageMinutes has 156
RoamingCalls has 156
PercChangeMinutes has 367
PercChangeRevenues has 367
ServiceArea has 24
Handsets has 1
HandsetModels has 1
CurrentEquipmentDays has 1
AgeHH1 has 909
AgeHH2 has 909
'''

data.select_dtypes(include = 'number').shape
# (51047, 35)

data.select_dtypes(include = 'object').shape
# (51047, 23)

# For now, only choose numeric features-
data_num = data.select_dtypes(include = 'number')

data_num.isna().values.any()
# True

data_num.isna().values.sum()
# 3491

# Replace all missing values with 0s-
data_num = data_num.replace(np.nan, 0)

data_num.isna().values.any()
# False

# Scale all numeric features using standard scaler-
std_scaler = StandardScaler()
data_num_std = std_scaler.fit_transform(data_num)

std_scaler.mean_.shape, std_scaler.scale_.shape
# ((35,), (35,))

# Scaled data has zero mean and unit variance-
data_num_std.mean(axis = 0)
'''
array([-4.98870681e-16, -2.02109439e-16, -6.68130376e-18, -1.22490569e-17,
        1.44761582e-17,  1.78168100e-17, -5.01097782e-18,  5.28936548e-18,
        4.73259017e-18,  2.33845632e-17, -2.22710125e-18,  1.01889882e-16,
       -5.01097782e-18,  1.39193828e-17, -9.46518033e-17,  3.03442546e-17,
        1.00219556e-16, -1.54783537e-16, -4.28716991e-17, -1.89303607e-17,
       -1.94871360e-18, -3.00658669e-17, -1.78168100e-17, -1.05787310e-17,
        6.73698129e-17, -1.33626075e-16, -1.42534480e-16,  3.89742719e-17,
        3.06226422e-17,  3.67471707e-17, -5.95749585e-17, -1.67032594e-18,
       -7.23807908e-18, -1.05787310e-16,  5.56775314e-17])
'''

data_num_std.std(axis = 0)
'''
array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
       1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
       1.])
'''

# data_num_std = data_num_std.astype(np.float32)


"""
Multi-core ML with scikit-learn:
scikit-learn Python machine learning library provides this capability via
the 'n_jobs' argument on key machine learning tasks, such as model training,
model evaluation, and hyperparameter tuning.

This configuration argument allows you to specify the number of cores to use
for the task. The default is None, which will use a single core. You can also
specify a number of cores as an integer, such as 1 or 2. Finally, you can
specify -1, in which case the task will use all of the cores available on your system.

n_jobs: Specify the number of cores to use for key machine learning tasks.

Common values are:

1. n_jobs=None: Use a single core or the default configured by your backend library.
2. n_jobs=4: Use the specified number of cores, in this case 4.
3. n_jobs=-1: Use all available cores.
"""

# Use DBSCAN algo with single core (default)-
model_dbscan = DBSCAN(
    eps = 0.5, min_samples = 5,
    metric = 'minkowski', metric_params = None,
    algorithm = 'auto', leaf_size = 30,
    p = 1, n_jobs = 1
)

start_time = time.time()
model_dbscan.fit(data_num_std)
end_time = time.time()
print(f"Training with 1 core took = {end_time - start_time:.2f} seconds")
# Training with 1 core took = 83.47 seconds


# Use DBSCAN algo with multiple cores-
model_dbscan = DBSCAN(
    eps = 0.5, min_samples = 5,
    metric = 'minkowski', metric_params = None,
    algorithm = 'auto', leaf_size = 30,
    p = 1, n_jobs = 4
)

start_time = time.time()
model_dbscan.fit(data_num_std)
end_time = time.time()
print(f"Training with 4 cores took = {end_time - start_time:.2f} seconds")
# Training with 4 cores took = 33.75 seconds


# Use DBSCAN algo with multiple cores-
model_dbscan = DBSCAN(
    eps = 0.5, min_samples = 5,
    metric = 'minkowski', metric_params = None,
    algorithm = 'auto', leaf_size = 30,
    p = 1, n_jobs = 6
)

start_time = time.time()
model_dbscan.fit(data_num_std)
end_time = time.time()
print(f"Training with 6 cores took = {end_time - start_time:.2f} seconds")
# Training with 6 cores took = 28.28 seconds


# Use DBSCAN algo with multiple cores-
model_dbscan = DBSCAN(
    eps = 0.5, min_samples = 5,
    metric = 'minkowski', metric_params = None,
    algorithm = 'auto', leaf_size = 30,
    p = 1, n_jobs = 8
)

start_time = time.time()
model_dbscan.fit(data_num_std)
end_time = time.time()
print(f"Training with 8 cores took = {end_time - start_time:.2f} seconds")
# Training with 8 cores took = 24.28 seconds

