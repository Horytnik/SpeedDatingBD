import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('Speed Dating Data.csv', engine='python')
data.head()

date_all_man = data.loc[data.gender == 1]


# Preparing our dataset

# %%

# Copy or assign?
# https://nedbatchelder.com/text/names.html
# https://pandas.pydata.org/pandas-docs/stable/getting_started/overview.html
# https://docs.python.org/3/faq/programming.html#why-did-changing-list-y-also-change-list-x
data_subset = data.loc[:, ['attr1_1', 'sinc1_1', 'intel1_1', 'fun1_1', 'amb1_1', 'shar1_1']].copy()

# %%

data_subset.head()
# Do you see something fishy here?

data_subset.iloc[:30, :]

data_subset.drop_duplicates(inplace=True)
data_subset.head()

# BTW: how many dates each participant had?
data.groupby('iid').count().id.unique()

# %%

data['iid'].value_counts().value_counts()
data.iid.unique().shape

data_subset.shape

data_subset = data.loc[:, ['iid', 'attr1_1', 'sinc1_1', 'intel1_1', 'fun1_1', 'amb1_1', 'shar1_1']].copy()
data_subset.drop_duplicates(inplace=True)
data_subset.drop(columns='iid', inplace=True)


data_subset

# K-means

# 4 samples/observations and 2 variables/features
data = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

scaler = StandardScaler()
data_subset_scaled = pd.DataFrame(scaler.fit_transform(data_subset))
#data_subset_scaled = data_subset

data_subset_scaled.min(axis=0)

print(data_subset.to_string())

data_subset_scaled.dropna(inplace=True)

data_subset.dropna(inplace=True)
data_subset_scaled = scaler.fit_transform(data_subset)
data_subset_scaled.mean(axis=0)  # You can also check var

kmeans = KMeans(n_clusters=4, init='k-means++', max_iter=300, n_init=10, random_state=0)
labels = kmeans.fit_predict(data_subset_scaled)


labels

# Principal Component Analysis
pca = PCA(n_components=2)
data_subset_pca = pca.fit_transform(data_subset_scaled)
data_subset_pca

data_subset_tf = pd.concat([pd.DataFrame(data_subset_pca, columns=['PC1', 'PC2']),
                            pd.DataFrame(labels, columns=['labels'])], axis=1)
data_subset_tf

data_subset_tf.plot.scatter(x='PC1', y='PC2', c='labels')

data_subset_tf.plot.scatter(x='PC1', y='PC2', c=data_subset_tf.labels.replace([0, 1, 2, 3], ['b', 'r', 'g', 'y']))

# Saving image
plot = data_subset_tf.plot.scatter(x='PC1', y='PC2',
                                   c=data_subset_tf.labels.replace([0, 1, 2, 3], ['b', 'r', 'g', 'y']))
fig = plot.get_figure()
fig.savefig("our_first_results.png", dpi=600)
pca.components_

np.matmul(pca.components_, np.array([10, 0, 0, 0, 0, 0]))

data_subset_tf.plot.scatter(x='PC1', y='PC2', c=data_subset_tf.labels.replace([0, 1, 2, 3], ['b', 'r', 'g', 'y']),
                            figsize=(10, 10))
# plt.arrow(0, 0, pca.components_[0,0]*6, pca.components_[0,1]*6, shape='left')
for i, colname in enumerate(data_subset.columns):
    plt.annotate(colname, ha='center', va='bottom', xy=(0, 0), size=15,
                 xytext=(pca.components_[0, i] * 4, pca.components_[1, i] * 4),
                 arrowprops={'arrowstyle': '<-'})
plt.show()

# Ok, but how many clusters?
#
# There
# are
# multiple
# methods, out
# of
# which
# the
# most
# popular
# are:
# - Elbow
# method
# - Silhouette(
# try at home)

# Elbow method taken from: https://medium.com/analytics-vidhya/how-to-determine-the-optimal-k-for-k-means-708505d204eb


def calculate_WSS(points, kmax):
    sse = []
    for k in range(1, kmax + 1):
        kmeans = KMeans(n_clusters=k).fit(points)
        centroids = kmeans.cluster_centers_
        pred_clusters = kmeans.predict(points)
        curr_sse = 0

        # calculate square of Euclidean distance of each point from its cluster center and add to current WSS
        for i in range(len(points)):
            curr_center = centroids[pred_clusters[i]]
            curr_sse += (points[i, 0] - curr_center[0]) ** 2 + (points[i, 1] - curr_center[1]) ** 2

        sse.append(curr_sse)
    return sse



plt.plot(np.linspace(1, 10, 10), calculate_WSS(data_subset_scaled, 10))
plt.xlabel('Number of clusters')
plt.ylabel('WSS')

plt.show()




