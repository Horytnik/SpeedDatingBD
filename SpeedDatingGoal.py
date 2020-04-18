import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
import sklearn
from sklearn.preprocessing import StandardScaler
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import Functions



# Data collection
data =  pd.read_csv("Speed Dating Data.csv", engine='python')

dataSubsetManList = Functions.getGoalsOfPeople(1, data)
dataSubsetWomanList = Functions.getGoalsOfPeople(0, data)


scaler = StandardScaler()
# dataSubsetScaledWoman = pd.DataFrame(scaler.fit_transform(dataSubsetWoman))
dataSubsetManList[1] = pd.DataFrame(scaler.fit_transform(dataSubsetManList[0]))

pca = PCA()
# data_Man_pca = pca.fit_transform(dataSubsetMan)
data_Woman_pca = pca.fit_transform(dataSubsetManList[1])

per_var = np.round(pca.explained_variance_ratio_ * 100, decimals=1)
labels = ['PC' + str(x) for x in range(1, len(per_var) + 1)]

plt.figure()
plt.bar(x=range(1, len(per_var) + 1), height=per_var, tick_label=labels)
plt.ylabel('Percentage of Explained Variance')
plt.xlabel('Principal Component')
plt.title('Scree Plot')

pca_df = pd.DataFrame(data_Woman_pca, columns=labels)

plt.figure()
plt.scatter(pca_df.PC1, pca_df.PC2)
plt.title('My PCA Graph')
plt.xlabel('PC1 - {0}%'.format(per_var[0]))
plt.ylabel('PC2 - {0}%'.format(per_var[1]))

for sample in pca_df.index:
    plt.annotate(sample, (pca_df.PC1.loc[sample], pca_df.PC2.loc[sample]))


dbscanData = Functions.calculateDbscanParm(1,50,2,20,data_Woman_pca)


dbscan = DBSCAN(eps = 1.9, min_samples = 2)
# dbscanMan = dbscan.fit(dataSubsetMan)
dbscanWoman = dbscan.fit(data_Woman_pca)

clusters = dbscanWoman.labels_

# kmeans = KMeans(n_clusters=5, init='k-means++', max_iter=300, n_init=10, random_state=0)
# labelsKMeans = kmeans.fit_predict(data_Woman_pca)


# colors = ['royalblue', 'maroon', 'forestgreen', 'mediumorchid', 'tan', 'deeppink', 'olive', 'goldenrod', 'lightcyan', 'navy', 'yellow','magneta', 'red', 'blue']
allcolors = ['C0','C1','C2','C3','C4','C5','C6','C7','C8','C9','C10','C11','C12','goldenrod']
colors = allcolors[:(max(clusters)+2)]
vectorizer = np.vectorize(lambda x: colors[x % len(colors)])

plt.figure()
plt.scatter(data_Woman_pca[:,0], data_Woman_pca[:,1], c=vectorizer(clusters))
# plt.scatter(data_Woman_pca[:,0], data_Woman_pca[:,1])
print(clusters)
plt.show()




data_subset_tf = pd.concat([pd.DataFrame(data_Woman_pca, columns=['PC1', 'PC2','PC3','PC4','PC5','PC6','PC7']),
                            pd.DataFrame(clusters, columns=['labels'])], axis=1)

data_subset_tf.plot.scatter(x='PC1', y='PC2', c=data_subset_tf.labels.replace([-1,0, 1, 2, 3], ['m','b', 'r', 'g', 'y']), figsize=(10,10))
#plt.arrow(0, 0, pca.components_[0,0]*6, pca.components_[0,1]*6, shape='left')
for i, colname in enumerate(dataSubsetManList[0].columns):
    plt.annotate(colname, ha='center', va='bottom', xy=(0, 0), size=15,
             xytext=(pca.components_[0,i]*4, pca.components_[1,i]*4),
             arrowprops = {'arrowstyle':'<-'})

