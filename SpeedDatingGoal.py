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

dbscan = DBSCAN(eps = 0.8, min_samples = 2)
dbscanMan = dbscan.fit(dataSubsetMan)
dbscanWoman = dbscan.fit(dataSubsetWoman)

outliers_df_Man = pd.DataFrame(dataSubsetMan)
print(Counter(dbscanMan.labels_))

#print(outliers_df_Man(dbscanMan.labels_ ==-1)),

figure = plt.figure()

ax = figure.add_axes([.1,.1,1,1])
colors = dbscanMan.labels_

ax.scatter( c=colors, s=120)

ax.set_xlabel('AAA')
#ax.set_y

