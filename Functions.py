import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN

def getGoalsOfPeople(gender, data):
    goalDict = {}
    dataSubsetList = []

    # Gender filtration
    dataSubsetGender = data.loc[data.gender == gender].copy()


    # Goal filtration
    for ctr in range(1, 7):
        dataSubset = dataSubsetGender.loc[dataSubsetGender.goal == ctr].copy()

        dataSubset = dataSubset.loc[:,
                        ['iid', 'attr1_1', 'sinc1_1', 'intel1_1', 'fun1_1', 'amb1_1', 'shar1_1']].copy()

        dataSubset = dataSubset.drop_duplicates()
        dataSubsetList.append(dataSubset.dropna())

        goalDict[ctr] = {len(dataSubset)}

    # Goals graph
    labels = ['Have fun', 'Meet new people', 'Have a date', 'Serious relationship', 'I did it!', 'Other']
    x = np.arange(len(labels))

    plt.figure()


    graph = plt.subplot()

    bar1 = graph.bar(0.5, goalDict[1], 0.5, label=labels[0])
    bar1 = graph.bar(1.5, goalDict[2], 0.5, label=labels[1])
    bar1 = graph.bar(2.5, goalDict[3], 0.5, label=labels[2])
    bar1 = graph.bar(3.5, goalDict[4], 0.5, label=labels[3])
    bar1 = graph.bar(4.5, goalDict[5], 0.5, label=labels[4])
    bar1 = graph.bar(5.5, goalDict[6], 0.5, label=labels[5])


    if gender == 1:
        graph.set_ylabel('Amount of men')
        graph.set_title('Amount of men who claim according goals')
    else:
        graph.set_ylabel('Amount of woman')
        graph.set_title('Amount of woman who claim according goals')

    graph.set_xticks(x + 0.5)
    graph.set_xticklabels(labels)
    plt.show()

    return dataSubsetList


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

def calculateDbscanParm(minEps,maxEps,minSampl,maxSampl, data):
    minEps = minEps*10
    maxEps = maxEps*10

    list = []

    for sampl in range(minSampl, maxSampl):
        for eps in range(minEps, maxEps, 1):
            dbscan = DBSCAN(eps=float(eps / 10), min_samples=sampl)

            dbscanWoman = dbscan.fit(data)

            clusters = dbscanWoman.labels_
            amountWrong = np.count_nonzero(clusters == -1)
            u, indices = np.unique(clusters, return_index=True)
            if len(u) >= 5:
                list.append([float(eps / 10), sampl, amountWrong])
            amountWrong = [row[2] for row in list]
            indexOfMin = amountWrong.index(min(amountWrong))

    return list[indexOfMin]
