# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 11:34:33 2017

@author: moazi
"""
import numpy as np
import itertools as itter
import sklearn.preprocessing as pre
import scipy.stats as stats
#Try ever which way to imporve A and D combination
# get data from csv
data = np.genfromtxt('AtoE.csv',  delimiter=',')
data = np.delete(data, (0), axis=0)  # remove labels columb
actual = np.genfromtxt('Actual.csv',  delimiter=',')
actual = np.delete(actual, (0), axis=0)
actual = actual[:, 2]  # 0 or 1 results col 2, ordered by index
data = np.vstack((data[:,0],data[:,1],data[:,4])).T#shape (720,2)

# input score array and actual ordered by subj trail, output precision
def precision(index, scores, true, score = True): # score = true , false = rank
    if score:
        r_or_s = -1
    else:
        r_or_s = 1
    numPos = 0    # number of positive cases
    for i in true:
        if i == 1:
            numPos += 1
    scores = scores.reshape(720, 1)
    index = index.reshape(720, 1)
    true = true.reshape(720, 1)
    scores = np.concatenate((index, scores), axis=1)  # add index col
    true = np.concatenate((index, true), axis=1)  # add index col
    scores = scores[scores[:, 1].argsort()[::r_or_s]]  #sort high to low when-1
    holdIndex = []
    for i in range(numPos):  # take indexes of the top 1 to numPos of scores
        holdIndex.append(scores[i, 0])
    correct = 0
    for i in holdIndex:  # looks up the index in the true set to see if correct
        index = np.where(true[:, 0] == i)
        y = true[index, 1]
        if y == 1:
            correct += 1
    return(correct/numPos)
    
def create_rank(index,scores):#given index and scores it returns a rank list
    table = np.vstack((index, scores)).T
    table = table[table[:,1].argsort()[::-1]]
    hold_same = []
    rank = []#hold ranks
    for i in range(len(table)-1):
        if table[i, 1] != table[i+1, 1]:
            if len(hold_same) > 0:
                hold_same.append(i+1)
                for j in hold_same:
                    rank.append(sum(hold_same)/(len(hold_same)))
                del hold_same[:]
            else:
                rank.append(i+1)
        else:
            hold_same.append(i+1)
    if len(hold_same) > 0: #special for the last row, it cant depend on next
        hold_same.append(720)
        for j in hold_same:
            rank.append(sum(hold_same)/(len(hold_same)))
        del hold_same[:]
    else:
        rank.append(720)
    order = np.hstack((table[:,0].reshape(720,1),np.array(rank).reshape(720,1)))
    order = order[order[:,0].argsort()]
    return order[:,1]

def mixGroupRank(ranks):
    shape = np.shape(np.array(ranks))  # rows [0], cols [1]
    comb_cols = []  # every combination ie) A,B,C,D,E,AB,AC,...,ABCDE
    mix_grp_r = []
    for r in range(1, shape[1]+1):
        for i in itter.combinations(range(0, shape[1]), r):
            comb_cols.append(i)
    for row in ranks:
        group_r = []
        for grp in comb_cols:
            minR = 100000  # min rank, set to high value out of range
            for i in grp:
                if row[i] < minR:
                    minR = row[i]
            group_r.append(minR)
        mix_grp_r.append(sum(group_r)/len(group_r))
    return mix_grp_r


# input scores, output average of highest score in every combo of A to E
def mixGroupScore(scores):
    shape = np.shape(np.array(scores))  # rows [0], cols [1]
    comb_cols = []  # every combination ie) A,B,C,D,E,AB,AC,...,ABCDE
    mix_grp_s = []
    for r in range(1, shape[1]+1):
        for i in itter.combinations(range(0, shape[1]), r):
            comb_cols.append(i)
    for row in scores:
        group_s = []
        for grp in comb_cols:
            minS = -1  # min rank, set to high value out of range
            for i in grp:
                if row[i] > minS:
                    minS = row[i]
            group_s.append(minS)
        mix_grp_s.append(sum(group_s)/len(group_s))
    return mix_grp_s


rankA = create_rank(data[:,0],data[:,1])
rankD = create_rank(data[:,0],data[:,2])
data = np.insert(data,3,[rankA,rankD], axis = 1)

initial_perf = np.array([[precision(data[:,0], data[:,1], actual),
            precision(data[:,0], data[:,2], actual)],
            [precision(data[:,0], data[:,3], actual, score = False),
            precision(data[:,0], data[:,4], actual, score = False)]])

perf_wght = initial_perf[0]/sum(initial_perf[0])
#print(np.shape())
averageScore = []
averageRank = []
minRank = []
maxScore = []
scoreComboByScorePerf = np.dot(perf_wght.reshape(1,2),data[:,1:3].T)
rankComboByScorePerf = np.dot(perf_wght.reshape(1,2),data[:,3:].T)
#scoreComboByRankPerf, rankComboByScorePerf = [],[]#redundent since same Perfm
for i in data:
    averageScore.append(np.average(i[1:3]))
    averageRank.append(np.average(i[3:]))
    maxScore.append(max(i[1:3]))
    minRank.append(min(i[3:]))
    
averageScore, averageRank = np.array(averageScore), np.array(averageRank)
minRank, maxScore = np.array(minRank),np.array(maxScore)
mixGrpRank = np.array(mixGroupRank(data[:,3:]))
mixGrpScore = np.array(mixGroupScore(data[:,1:3]))




#Acuracies
print('A and E score perf and rank perf\n', initial_perf,'\n\n')
print('average score perf\n',precision(data[:,0], averageScore, actual))
print('Average rank perf (rank converted to score)\n',
      precision(data[:,0], 1/averageRank, actual))#same as next
print('Average rank perf \n', precision(data[:,0], averageRank, actual, score = False))
print('Max score perf \n', precision(data[:,0], maxScore, actual))
print('Min rank perf (rank converted to score)\n',
      precision(data[:,0], (1/minRank), actual))#better then next
print('Min rank perf \n',precision(data[:,0], minRank, actual, score = False))
print('Mix Group score perf\n', precision(data[:,0], mixGrpScore, actual))
print('Mix Group rank perf (rank converted to score)\n',precision(data[:,0], 1/mixGrpRank, actual))
print('Mix Group rank perf \n',precision(data[:,0], mixGrpRank, actual, score = False))
print('Score combo Weighted by perf\n', precision(data[:,0], scoreComboByScorePerf, actual))
print('Rank combo Weighted by perf (rank converted to score)\n', precision(data[:,0], 1/rankComboByScorePerf, actual))
print('Rank combo Weighted by perf \n',precision(data[:,0], rankComboByScorePerf, actual, score = False))
