# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 20:35:15 2017

@author: moazi
"""
import numpy as np
import scipy.stats as stats
import sklearn.preprocessing as pre
import itertools as itter


def precision(index, scores, true):  # input numpy arrays all in same order
    numPos = 0    # number of positive cases
    for i in true:
        if i == 1:
            numPos += 1
    scores = scores.reshape(len(scores), 1)
    index = index.reshape(len(scores), 1)
    true = true.reshape(len(scores), 1)
    scores = np.concatenate((index, scores), axis=1)  # add index col
    true = np.concatenate((index, true), axis=1)  # add index col
    scores = scores[scores[:, 1].argsort()[::-1]]  # sort higest to lowest
    holdIndex = []
    for i in range(numPos):  # take indexes of the top 1 to numPos of scores
        holdIndex.append(scores[i, 0])
    correct = 0
    for i in holdIndex:  # looks up the index in the true set to see if correct
        index = np.where(true[:, 0] == i)
        y = true[index, 1]
        if y == 1:
            correct += 1
    print(100*(correct/numPos), "%")
    holdIndex = []
    for i in range(300):  # take indexes of the top 1 to numPos of scores
        holdIndex.append(scores[i, 0])
    correct = 0
    for i in holdIndex:  # looks up the index in the true set to see if correct
        index = np.where(true[:, 0] == i)
        y = true[index, 1]
        if y == 1:
            correct += 1
    print(100*(correct/300), "%")
    holdIndex = []
    for i in range(200):  # take indexes of the top 1 to numPos of scores
        holdIndex.append(scores[i, 0])
    correct = 0
    for i in holdIndex:  # looks up the index in the true set to see if correct
        index = np.where(true[:, 0] == i)
        y = true[index, 1]
        if y == 1:
            correct += 1
    print(100*(correct/200), "%")
    holdIndex = []
    for i in range(100):  # take indexes of the top 1 to numPos of scores
        holdIndex.append(scores[i, 0])
    correct = 0
    for i in holdIndex:  # looks up the index in the true set to see if correct
        index = np.where(true[:, 0] == i)
        y = true[index, 1]
        if y == 1:
            correct += 1
    print(100*(correct/100), "%")


def checkPerGroupAvergeScore(index, scores, true):  # matrix,cols=score systm
    shape = np.shape(np.array(scores))  # rows [0], cols [1]
    comb_cols = []  # index val of every combo ie) A,B,C,D,E,AB,AC,...,ABCDE
    for r in range(1, shape[1]+1):  # number of cols
        for i in itter.combinations(range(0, shape[1]), r):
            comb_cols.append(i)
    for grp in comb_cols:
        print("testing combo", grp)
        holdMod = []
        for each in grp:
            holdMod.append(scores[:, each])
        holdMod = np.array(holdMod)
        try:
            model = np.average(holdMod.T, axis=1)
        except:
            model = holdMod.T
        precision(index, model.reshape(720, 1), true)


# Main _______
data = np.genfromtxt('AtoE.csv',  delimiter=',')
data = np.delete(data, (0), axis=0)  # remove labels columb
actual = np.genfromtxt('Actual.csv',  delimiter=',')
actual = np.delete(actual, (0), axis=0)
actual = actual[:, 2]  # 0 or 1 results col 2, ordered by index

for i in range(1, 6):
    print("model", i)
    precision(data[:, 0], data[:, i], actual)


checkPerGroupAvergeScore(data[:, 0], data[:, 1:], actual)

"""
RESULTS:
    From this test We found that
    3 combo out performed at every percision 
    testing combo (0, 2), (0, 4), (0, 2, 3)
    4 more combination of these models outperformed or
    matched at every percision (0,1),(0, 1, 4),(0, 2, 4),(0, 3, 4)
    
"""
"""
sort_by = data

for i in range(1, 6):
    sort_by = sort_by[sort_by[:, i].argsort()[::-1]]
    rank = np.array(ranking(sort_by, i)).reshape(720, 1)
    sort_by = np.hstack([sort_by, rank])
    sort_by = sort_by[sort_by[:, 0].argsort()]

scoresNDranks = sort_by  # array with index ranks and scores
scores = data[:, 1:6]  # just scores Ato E
ranks = scoresNDranks[:, 6:11]  # just Ranks A to E

checkPerGroupAvergeRanks(data[:,0], ranks, actual)
"""
"""
RESULTS:
From this test We found that
0 models outperformed or matched at #pos and P@300
and the same models outperformed or matched a at 200, and 100
Rank Performance varied since averaging ranks (especially that of A, would heavily skew results)
"""