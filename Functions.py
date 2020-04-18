import numpy as np
import matplotlib.pyplot as plt

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
        dataSubsetList.append(dataSubset)

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
