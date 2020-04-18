import pandas as pd
from sklearn.cluster import DBSCAN
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

dataSubsetMan = Functions.getGoalsOfPeople(1, data)
dataSubsetWoman = Functions.getGoalsOfPeople(0, data)


dataSubsetMan.dropna(inplace=True)
dataSubsetWoman.dropna(inplace=True)


# scaler = StandardScaler()
# dataSubsetScaledWoman = pd.DataFrame(scaler.fit_transform(dataSubsetWoman))
# dataSubsetScaledMan = pd.DataFrame(scaler.fit_transform(dataSubsetMan))

pca = PCA()
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

# colors = ['royalblue', 'maroon', 'forestgreen', 'mediumorchid', 'tan', 'deeppink', 'olive', 'goldenrod', 'lightcyan', 'navy', 'yellow','magneta', 'red', 'blue']
colors = ['C0','C1','C2','C3','C4','C5','C6','C7','C8','C9','C10','C11']
vectorizer = np.vectorize(lambda x: colors[x % len(colors)])

plt.scatter(data_Woman_pca[:,0], data_Woman_pca[:,1], c=vectorizer(clusters))
# plt.scatter(data_Woman_pca[:,0], data_Woman_pca[:,1])
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

