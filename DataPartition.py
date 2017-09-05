import pandas as pd
import numpy as np
import random
from sklearn import metrics
import matplotlib.pyplot as plt
from collections import Counter

from MasterRS import *
from CollaborativeFilter import *

class DataPartition:
    def __init__(self, consolidatedDF, colItems, colUsers, partitionRatios, testOrValid):
        self.colItems = colItems
        self.colUsers = colUsers
        self.consolidatedDF = consolidatedDF
        self.testOrValid = testOrValid
        self.trainDF, self.validDF, self.testDF = self.partitionDataset(consolidatedDF, partitionRatios)
        self.userItemDict = self.makeUserItemDict()

    # Returns dataframe where random proportion = proportionTrain of each user's items have their userIDs hidden
    def partitionDataset(self, dataframe, partitionRatios):
        userList = list(set(self.consolidatedDF[self.colUsers]))
        assert(sum(partitionRatios) == 1)

        pTrain = partitionRatios[0]
        pValid = partitionRatios[1]
        trainList = []
        validList = []
        testList = []

        for user in userList:
            itemsDetails = dataframe[dataframe[self.colUsers] == user]
            nItems = len(itemsDetails)

            nTrain = int(np.floor(pTrain * nItems))
            nValidation = int(np.ceil(pValid * nItems))

            permItemDetails = itemsDetails.sample(frac=1, random_state=1)
            trainList.append(permItemDetails.iloc[:nTrain, :])
            validList.append(permItemDetails.iloc[nTrain:(nTrain+nValidation), :])
            testList.append(permItemDetails.iloc[(nTrain+nValidation):, :])

        trainDF = pd.concat(trainList)
        validDF = pd.concat(validList)
        testDF = pd.concat(testList)

        return trainDF, validDF, testDF


    def coldPartition(self, coldRatio):
        newTrainDF, newTestDF = self.getNewTrainTestSet()
        trainItemsList = list(newTrainDF[self.colItems])

        itemCountDict = Counter(trainItemsList)
        coldLimitDict = {}

        for key, value in itemCountDict.items():
            coldLimitDict[key] = int(value * coldRatio)

        coldTrainList = []
        for item in set(trainItemsList):
            usersViewed = newTrainDF[newTrainDF[self.colItems] == item]
            nColdLimit = coldLimitDict[item]
            coldTrainList.append(usersViewed.iloc[:nColdLimit, :])

        coldTrainDF = pd.concat(coldTrainList)
        return coldTrainDF, newTestDF

    def getNewTrainTestSet(self):
        if self.testOrValid == 'test':
            newTrainDF = self.trainDF.append(self.validDF)
            newTestDF = self.testDF.copy()
        elif self.testOrValid == 'valid':
            newTrainDF = self.trainDF.copy()
            newTestDF = self.validDF.copy()
        return newTrainDF, newTestDF

    def getAllTrainItems(self, newTrainDF):
        trainItems = set(list(newTrainDF[self.colItems]))
        return trainItems

    def getTestItems(self, user, newTrainDF):
        userViewedItems = self.userItemDict[user]
        itemDetailsTrain = newTrainDF[newTrainDF[self.colUsers] == user]
        userTrainItems = itemDetailsTrain[self.colItems]
        trainItems = self.getAllTrainItems(newTrainDF)
        hiddenItems = list(set(userViewedItems) - set(userTrainItems))
        testItems = [item for item in hiddenItems if item in trainItems]
        return testItems

    # Constructs dictionary where keys are users and values are list of items they have viewed (none hidden)
    def makeUserItemDict(self):

        if self.testOrValid == 'test':
            newAllDF = self.consolidatedDF.copy()
        elif self.testOrValid == 'valid':
            newAllDF = self.trainDF.append(self.validDF)

        userList = list(set(self.consolidatedDF[self.colUsers]))
        userItemDict = {}
        for user in userList:
            itemsDetails = newAllDF[newAllDF[self.colUsers] == user]
            itemsViewed = list(itemsDetails[self.colItems])
            userItemDict[user] = itemsViewed

        return userItemDict