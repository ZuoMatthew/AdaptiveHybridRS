import numpy as np
import pandas as pd
import scipy
from scipy import spatial

class CollaborativeFilter:
    def __init__(self, dataframe, colItems, colUsers):
        self.dataframe = dataframe
        self.colItems = colItems
        self.colUsers = colUsers
        self.cleanDF = self.cleanDataframe()

    # Simplify dataframe to only contain users and items that they have viewed, and remove duplicates
    def cleanDataframe(self):
        itemUserDF = self.dataframe[[self.colItems, self.colUsers]]
        return itemUserDF.drop_duplicates()

    # Gets list of all unique users from filtered criteria
    def getUserList(self):
        userList = list(set(self.cleanDF[self.colUsers]))
        return userList

    # Gets list of all unique items from filtered criteria
    def getItemList(self):
        itemUserDict = self.cleanDF.groupby(self.colItems)[self.colUsers].apply(list).to_dict()
        itemList = [item for item in itemUserDict]
        return itemList

    # Builds a matrix denoting the items and which users have viewed them
    # Each row is a unique user and a binary 0-1 entry for whether a user (column) has
    # viewed the item or not
    def buildItemUserMatrix(self):
        userList = self.getUserList()
        itemList = self.getItemList()

        itemUserDict = self.cleanDF.groupby(self.colItems)[self.colUsers].apply(list).to_dict()
        userIndexDict = dict(zip(userList, range(len(userList))))

        itemUserMatrix = []
        nUsers = len(userList)
        for item in itemList:
            itemVector = np.zeros(nUsers, dtype=np.int)
            userIdList = itemUserDict.get(item)
            userIndices = [userIndexDict.get(user) for user in userIdList]
            itemVector[userIndices] = 1
            itemUserMatrix.append(itemVector)
        return itemUserMatrix, itemList, userList

    # Constructs a square similarity matrix for similarities between all items
    def buildSimilarityMatrix(self, itemUserMatrix, metric="cosine"):
        nItems = len(itemUserMatrix)
        simMatrix = np.zeros([nItems, nItems])

        for i in range(nItems):
            for j in range(nItems):
                itemViews1 = itemUserMatrix[i]
                itemViews2 = itemUserMatrix[j]

                if metric == "cosine":
                    itemSimilarity = self.calculateCosineSimilarity(itemViews1, itemViews2)
                elif metric == "pearson":
                    itemSimilarity = self.calculatePearsonCorrelation(itemViews1, itemViews2)
                elif metric == "spearman":
                    itemSimilarity = self.calculateSpearmanRankCorrelation(itemViews1, itemViews2)
                elif metric == "euclidean":
                    itemSimilarity = self.calculateEuclideanSimilarity(itemViews1, itemViews2)
                else:
                    print("# Error: Invalid similarity metric specified #")
                simMatrix[i][j] = itemSimilarity
            if i % 100 == 0: print("Sim matrix progress: ", i, "/", nItems)

        return simMatrix

    # Generates a list of recommendations for a user
    # Proposed list of items are most similar to the items that the user previously viewed
    def makeRecommendation(self, user, itemList, similarityMatrix, nRecommend):
        viewedItems = self.cleanDF[self.cleanDF[self.colUsers] == user][self.colItems]
        viewedIndices = [itemList.index(vi) for vi in viewedItems] # Finds index of viewed items in itemList
        simRows = [similarityMatrix[vid] for vid in viewedIndices]
        colSums = np.sum(simRows, 0)

        for vi in sorted(viewedIndices, reverse=True):
            colSums[vi] = 0

        nearestItemIndices = np.argsort(-colSums)
        recomItemIndices = nearestItemIndices[:nRecommend]
        recomItems = [itemList[i] for i in recomItemIndices]

        return recomItems

    def createRecommenderDict(self, userList, itemList, similarityMatrix, nRecommend):
        recommenderDict = {}
        for user in userList:
            recomItems = self.makeRecommendation(user, itemList, similarityMatrix, nRecommend)
            recommenderDict[user] = recomItems

        return recommenderDict


    def calculateCosineSimilarity(self, vec1, vec2):
        cosDistance = spatial.distance.cosine(vec1, vec2)
        itemSimilarity = 1 - cosDistance
        return itemSimilarity


    def calculatePearsonCorrelation(self, vec1, vec2):
        pearsonCorrelation = scipy.stats.pearsonr(vec1, vec2)
        return pearsonCorrelation[0]


    def calculateSpearmanRankCorrelation(self, vec1, vec2):
        spearmanCorrelation = scipy.stats.spearmanr(vec1, vec2)
        return spearmanCorrelation[0]


    def calculateEuclideanSimilarity(self, vec1, vec2):
        euclideanDistance = spatial.distance.euclidean(vec1, vec2)
        itemSimilarity = 1 - euclideanDistance
        return itemSimilarity
