import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('Speed Dating Data.csv', engine='python')
data.head()

data_all_man = data.loc[data.gender == 1]
data_all_woman = data.loc[data.gender == 0]

data_subset_man = data_all_man.loc[:, ['attr1_1', 'sinc1_1', 'intel1_1', 'fun1_1', 'amb1_1', 'shar1_1']].copy()
data_subset_woman = data_all_woman.loc[:, ['attr1_1', 'sinc1_1', 'intel1_1', 'fun1_1', 'amb1_1', 'shar1_1']].copy()

data_subset_man.dropna(inplace=True)
data_subset_woman.dropna(inplace=True)

kmeans = KMeans(n_clusters=5, init='k-means++', max_iter=300, n_init=10, random_state=0)
labels_woman = kmeans.fit_predict(data_subset_woman)
labels_man = kmeans.fit_predict(data_subset_man)

scaler = StandardScaler()
data_subset_woman_scaled = pd.DataFrame(scaler.fit_transform(data_subset_woman))
data_subset_man_scaled = pd.DataFrame(scaler.fit_transform(data_subset_man))


# PCA WOMAN

pca = PCA(n_components=2)
data_subset_pca = pca.fit_transform(data_subset_woman_scaled)
data_subset_pca

data_subset_tf = pd.concat([pd.DataFrame(data_subset_pca, columns=['PC1', 'PC2']),
                            pd.DataFrame(labels_woman, columns=['labels'])], axis=1)
data_subset_tf

data_subset_tf.plot.scatter(x='PC1', y='PC2', c='labels')

data_subset_tf.plot.scatter(x='PC1', y='PC2', c=data_subset_tf.labels.replace([0, 1, 2, 3, 4], ['b', 'r', 'g', 'y','m']))

# Saving image
plot = data_subset_tf.plot.scatter(x='PC1', y='PC2',
                                   c=data_subset_tf.labels.replace([0, 1, 2, 3, 4], ['b', 'r', 'g', 'y','m']))
fig = plot.get_figure()
fig.savefig("our_first_results.png", dpi=600)
pca.components_

np.matmul(pca.components_, np.array([10, 0, 0, 0, 0, 0]))

data_subset_tf.plot.scatter(x='PC1', y='PC2', c=data_subset_tf.labels.replace([0, 1, 2, 3, 4], ['b', 'r', 'g', 'y','m']),
                            figsize=(10, 10))
# plt.arrow(0, 0, pca.components_[0,0]*6, pca.components_[0,1]*6, shape='left')
for i, colname in enumerate(data_subset_woman.columns):
    plt.annotate(colname, ha='center', va='bottom', xy=(0, 0), size=15,
                 xytext=(pca.components_[0, i] * 4, pca.components_[1, i] * 4),
                 arrowprops={'arrowstyle': '<-'})
plt.show()

# PCA MAN

pca = PCA(n_components=2)
data_subset_pca = pca.fit_transform(data_subset_man_scaled)
data_subset_pca

data_subset_tf = pd.concat([pd.DataFrame(data_subset_pca, columns=['PC1', 'PC2']),
                            pd.DataFrame(labels_man, columns=['labels'])], axis=1)
data_subset_tf

data_subset_tf.plot.scatter(x='PC1', y='PC2', c='labels')

data_subset_tf.plot.scatter(x='PC1', y='PC2', c=data_subset_tf.labels.replace([0, 1, 2, 3, 4], ['b', 'r', 'g', 'y','m']))

# Saving image
plot = data_subset_tf.plot.scatter(x='PC1', y='PC2',
                                   c=data_subset_tf.labels.replace([0, 1, 2, 3, 4], ['b', 'r', 'g', 'y','m']))
fig = plot.get_figure()
fig.savefig("our_first_results.png", dpi=600)
pca.components_

np.matmul(pca.components_, np.array([10, 0, 0, 0, 0, 0]))

data_subset_tf.plot.scatter(x='PC1', y='PC2', c=data_subset_tf.labels.replace([0, 1, 2, 3, 4], ['b', 'r', 'g', 'y','m']),
                            figsize=(10, 10))
# plt.arrow(0, 0, pca.components_[0,0]*6, pca.components_[0,1]*6, shape='left')
for i, colname in enumerate(data_subset_man.columns):
    plt.annotate(colname, ha='center', va='bottom', xy=(0, 0), size=15,
                 xytext=(pca.components_[0, i] * 4, pca.components_[1, i] * 4),
                 arrowprops={'arrowstyle': '<-'})
plt.show()