import pandas as pd
from sklearn.cluster import DBSCAN
import sklearn
from sklearn.preprocessing import StandardScaler

from collections import Counter

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import numpy as np
import matplotlib.pyplot as plt

data =  pd.read_csv("Speed Dating Data.csv", engine='python')

dataMan = data.loc[data.gender == 0].copy()
dataWoman = data.loc[data.gender == 1].copy()

dataSubsetMan = dataMan.loc[dataMan.goal == 4].copy()
dataSubsetWoman = dataWoman.loc[dataWoman.goal == 4].copy()

dataSubsetMan = dataSubsetMan.loc[:,['samerace','age_o','attr1_1','sinc1_1','intel1_1','fun1_1','amb1_1','shar1_1']].copy()
dataSubsetWoman = dataSubsetWoman.loc[:,['samerace','age_o','attr1_1','sinc1_1','intel1_1','fun1_1','amb1_1','shar1_1']].copy()

dataSubsetMan.dropna(inplace=True)
dataSubsetWoman.dropna(inplace=True)

# scaler = StandardScaler()
# dataSubsetScaledWoman = pd.DataFrame(scaler.fit_transform(dataSubsetWoman))
# dataSubsetScaledMan = pd.DataFrame(scaler.fit_transform(dataSubsetMan))

pca = PCA(n_components=2)
# data_Man_pca = pca.fit_transform(dataSubsetMan)
data_Woman_pca = pca.fit_transform(dataSubsetWoman)

per_var = np.round(pca.explained_variance_ratio_ * 100, decimals=1)
labels = ['PC' + str(x) for x in range(1, len(per_var) + 1)]

plt.bar(x=range(1, len(per_var) + 1), height=per_var, tick_label=labels)
plt.ylabel('Percentage of Explained Variance')
plt.xlabel('Principal Component')
plt.title('Scree Plot')
plt.show()

pca_df = pd.DataFrame(data_Woman_pca, columns=labels)

plt.scatter(pca_df.PC1, pca_df.PC2)
plt.title('My PCA Graph')
plt.xlabel('PC1 - {0}%'.format(per_var[0]))
plt.ylabel('PC2 - {0}%'.format(per_var[1]))

for sample in pca_df.index:
    plt.annotate(sample, (pca_df.PC1.loc[sample], pca_df.PC2.loc[sample]))

plt.show()

dbscan = DBSCAN(eps = 0.8, min_samples = 2)
# dbscanMan = dbscan.fit(dataSubsetMan)
dbscanWoman = dbscan.fit(data_Woman_pca)

clusters = dbscanWoman.labels_

colors = ['royalblue', 'maroon', 'forestgreen', 'mediumorchid', 'tan', 'deeppink', 'olive', 'goldenrod', 'lightcyan', 'navy']
vectorizer = np.vectorize(lambda x: colors[x % len(colors)])

plt.scatter(dbscanWoman[:,0], dbscanWoman[:,1], c=vectorizer(clusters))
plt.show()


outliers_df_Woman = pd.DataFrame(data_Woman_pca)
# print(Counter(dbscanMan.labels_))

#print(outliers_df_Man(dbscanMan.labels_ ==-1)),

figure = plt.figure()

ax = figure.add_axes([.1,.1,1,1])
# colors = dbscanMan.labels_

ax.scatter( c=colors, s=120)

ax.set_xlabel('AAA')
#ax.set_y

