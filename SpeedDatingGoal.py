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


for genderSel in range(0,2):
    dataSubsetList = Functions.getGoalsOfPeople(genderSel, data)
    for goals in range(0,2):
        scaler = StandardScaler()
        # dataSubsetScaledWoman = pd.DataFrame(scaler.fit_transform(dataSubsetWoman))
        dataSubsetList[goals] = pd.DataFrame(scaler.fit_transform(dataSubsetList[goals]))

        # dataSubsetList[goals].drop(columns='iid', inplace=True)
        #
        pca = PCA()
        # data_Man_pca = pca.fit_transform(dataSubsetMan)
        data_Man_pca = pca.fit_transform(dataSubsetList[goals])

        per_var = np.round(pca.explained_variance_ratio_ * 100, decimals=1)
        labels = ['PC' + str(x) for x in range(1, len(per_var) + 1)]

        plt.figure()
        plt.bar(x=range(1, len(per_var) + 1), height=per_var, tick_label=labels)
        plt.ylabel('Percentage of Explained Variance')
        plt.xlabel('Principal Component')
        plt.title('Scree Plot')
        if genderSel == 1:
            figName = ".\ResultFig\Scree Plot Man goal={}".format(goals)
            plt.savefig(figName)
        else:
            figName = ".\ResultFig\Scree Plot Woman goal={}".format(goals)
            plt.savefig(figName)


        pca_df = pd.DataFrame(data_Man_pca, columns=labels)

        plt.figure()
        plt.scatter(pca_df.PC1, pca_df.PC2)
        plt.title('My PCA Graph')
        plt.xlabel('PC1 - {0}%'.format(per_var[0]))
        plt.ylabel('PC2 - {0}%'.format(per_var[1]))

        for sample in pca_df.index:
            plt.annotate(sample, (pca_df.PC1.loc[sample], pca_df.PC2.loc[sample]))
        if genderSel == 1:
            figName = ".\ResultFig\PCA Man goal={}".format(goals)
            plt.savefig(figName)
        else:
            figName = ".\ResultFig\PCA Woman goal={}".format(goals)
            plt.savefig(figName)
        dbscanData = Functions.calculateDbscanParm(1,50,2,20,data_Man_pca)


        dbscan = DBSCAN(eps = 1.9, min_samples = 2)
        # dbscanMan = dbscan.fit(dataSubsetMan)
        dbscanWoman = dbscan.fit(data_Man_pca)

        clusters = dbscanWoman.labels_

        allcolors = ['C0','C1','C2','C3','C4','C5','C6','C7','C8','C9','C10','C11','C12','goldenrod']
        colors = allcolors[:(max(clusters)+2)]
        vectorizer = np.vectorize(lambda x: colors[x % len(colors)])

        plt.figure()
        plt.scatter(data_Man_pca[:,0], data_Man_pca[:,1], c=vectorizer(clusters))
        plt.title("DBSCAN Clustering")
        if genderSel == 1:
            figName = ".\ResultFig\DBSCAN Man goal={}".format(goals)
            plt.savefig(figName)
        else:
            figName = ".\ResultFig\DBSCAN Woman goal={}".format(goals)
            plt.savefig(figName)
        # plt.show()




        data_subset_tf = pd.concat([pd.DataFrame(data_Man_pca, columns=['PC1', 'PC2','PC3','PC4','PC5','PC6']),
                                    pd.DataFrame(clusters, columns=['labels'])], axis=1)

        unigueClust, indices = np.unique(clusters, return_index=True)

        data_subset_tf.plot.scatter(x='PC1', y='PC2', c=data_subset_tf.labels.replace(unigueClust, colors), figsize=(10,10))
        #plt.arrow(0, 0, pca.components_[0,0]*6, pca.components_[0,1]*6, shape='left')
        for i, colname in enumerate(dataSubsetList[goals+1].columns):
            plt.annotate(colname, ha='center', va='bottom', xy=(0, 0), size=15,
                     xytext=(pca.components_[0,i]*4, pca.components_[1,i]*4),
                     arrowprops = {'arrowstyle':'<-'})
        if genderSel == 1:
            figName = ".\ResultFig\Arrow Man goal={}".format(goals)
            plt.savefig(figName)
        else:
            figName = ".\ResultFig\Arrow Woman goal={}".format(goals)
            plt.savefig(figName)