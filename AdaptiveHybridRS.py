import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from MasterRS import *
from CollaborativeFilter import *

class AdaptiveHybridRS:
    def __init__(self, recommendedCBFDict, recommendedCFDict, trainTestDF):
        self.recommendedCBFDict = recommendedCBFDict
        self.recommendedCFDict = recommendedCFDict
        self.trainTestDF = trainTestDF


    def weightedCombine(self, cbfWeight, cfWeight):
        userList = [user for user in self.recommendedCBFDict.keys() if user in self.recommendedCFDict]
        combinedRecDict = {}

        for user in userList:
            usersItemsCBF = self.recommendedCBFDict[user]
            usersItemsCF = self.recommendedCFDict[user]
            itemList = [item for item in usersItemsCBF if item in usersItemsCF]

            combinedWeights = []
            for item in itemList:
                itemRankCBF = usersItemsCBF.index(item)
                itemRankCF = usersItemsCF.index(item)
                weightedRank = cbfWeight*itemRankCBF + cfWeight*itemRankCF
                combinedWeights.append(weightedRank)

            bestItemIndices = np.argsort(np.array(combinedWeights))
            recomItems = [usersItemsCBF[i] for i in bestItemIndices]
            combinedRecDict[user] = recomItems

        return combinedRecDict

