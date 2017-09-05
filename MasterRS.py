import pandas as pd
import numpy as np
from PropertyData import *
from CollaborativeFilter import *
from ContentBasedFilter import *

class MasterRS:

    def __init__(self, filename, chunkSize, colItems, colUsers, colNames, nSharedItems, nSharedUsers, nIter, colTypes):
        # self.fullData = self.getFullData(filename)
        self.chunkSize = chunkSize
        self.colItems = colItems
        self.colUsers = colUsers
        self.colNames = colNames
        self.nSharedItems = nSharedItems
        self.nSharedUsers = nSharedUsers
        self.nIter = nIter
        self.colTypes = colTypes

    # Takes a csv file and consolidates it into a pandas dataframe with rare occurernces and anomalies removed
    def consolidateData(self, filename):
        prda = PropertyData(self.colNames, self.chunkSize, self.colItems, self.colUsers)
        fullData = prda.csvToPandas(filename)
        simplifiedData = prda.removeDuplicates(fullData)
        reducedData = prda.alternateRemoveSparse(simplifiedData, self.colItems, self.colUsers,
                                                 self.nSharedItems, self.nSharedUsers, self.nIter)

        print("No. frequent users = ", len(set(reducedData[self.colUsers])))
        print("No. frequent items = ", len(set(reducedData[self.colItems])), "\n")
        return reducedData

    def getUserList(self, dataframe):
        cf = CollaborativeFilter(dataframe, self.colItems, self.colUsers)
        return cf.getUserList()

    def getItemList(self, dataframe):
        cf = CollaborativeFilter(dataframe, self.colItems, self.colUsers)
        return cf.getItemList()


    def getCF(self, dataframe):
        cf = CollaborativeFilter(dataframe, self.colItems, self.colUsers)
        return cf

    def getCBF(self, dataframe, kernel):
        colCategorical = self.colTypes[0]
        colContinuous = self.colTypes[1]
        colDiscrete = self.colTypes[2]
        cbf = ContentBasedFilter(dataframe, self.colItems, self.colUsers,
                                 colContinuous, colCategorical, colDiscrete, kernel)
        return cbf


    # Constructs the similarity matrix from the Collaborative Filtering section
    def getCollaborativeSimMatrix(self, dataframe, metric):
        cf = self.getCF(dataframe)
        squishedMatrix, itemList, userList = cf.buildItemUserMatrix()
        simMatrix = cf.buildSimilarityMatrix(squishedMatrix, metric)
        return simMatrix

    # Performs the collaborative filtering for recommendations given a single user
    def collaborativeRecommendation(self, dataframe, user, simMatrix, nRecom):
        cf = self.getCF(dataframe)
        itemList = self.getItemList(dataframe)
        recomItemsCF = cf.makeRecommendation(user, itemList, simMatrix, nRecom)
        return recomItemsCF

    def contentBasedRecommendation(self, dataframe, user, nRecom, kernel):
        cbf = self.getCBF(dataframe, kernel)
        recomItemsCBF = cbf.makeRecommendation(user, nRecom)
        return recomItemsCBF


    def getCFRecommenderDict(self, dataframe, simMatrix, nRecom):
        cf = self.getCF(dataframe)
        userList = self.getUserList(dataframe)
        itemList = self.getItemList(dataframe)
        recDictCF = cf.createRecommenderDict(userList, itemList, simMatrix, nRecom)
        return recDictCF


    def getCBFRecommenderDict(self, dataframe, kernel, nRecom):
        cbf = self.getCBF(dataframe, kernel)
        userList = self.getUserList(dataframe)
        recDictCBF = cbf.createRecommenderDict(userList, nRecom)
        return recDictCBF