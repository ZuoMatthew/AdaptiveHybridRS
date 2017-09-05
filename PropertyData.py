import numpy as np
import pandas as pd
from collections import Counter
import os

class PropertyData:
    def __init__(self, colNames, chunkSize, colItems, colUsers):
        self.colNames = colNames
        self.chunkSize = chunkSize
        self.colItems = colItems
        self.colUsers = colUsers

    # Transforms .csv file to pandas format in Python
    def csvToPandas(self, filename):
        i=0
        arrayDF = []
        for chunk in pd.read_csv(filename, header=None, index_col=False, chunksize=self.chunkSize):
            simplifiedData = self.removeDuplicates(chunk.iloc[:, self.colNames])
            correctUsersData = self.removeNoneUsers(simplifiedData, self.colUsers)
            arrayDF.append(correctUsersData)
            print(i)
            i += 1

        df = pd.concat(arrayDF)
        # print(df.head(10))
        return df

    # Separates a dataframe into n separate dataframes provided in the 'ratios' array
    def partitionDataframe(self, dataframe, ratios):
        assert sum(ratios) == 1
        splitCumul = np.cumsum(ratios)[:-1]
        splitIndex = [int(np.floor(sc * len(dataframe))) for sc in splitCumul]
        splitDataframes = np.split(dataframe, splitIndex)
        return splitDataframes

    def removeDuplicates(self, dataframe):
        return dataframe.drop_duplicates()

    def removeNoneUsers(self, dataframe, colUsers):
        errorName = "none"
        unknownName = "unknown"
        removedNoneDF = dataframe[dataframe[colUsers] != errorName]
        return removedNoneDF[removedNoneDF[colUsers] != unknownName]


    def orderByCol(self, dataframe, col):
        return dataframe.sort_values([col])

    # Removes rows in the dataframe where the column entry 'col' for that row appears fewer times than
    # 'threshold' in the dataframe
    def removeSparseEntries(self, dataframe, threshold, col):
        truncatedDF = dataframe.groupby(col).filter(lambda x: len(x) >= threshold)
        return truncatedDF

    def alternateRemoveSparse(self, dataframe, col1, col2, threshold1, threshold2, nIter):
        iteratedDF = dataframe.copy()
        for i in range(nIter):
            print(len(iteratedDF))
            iteratedDF = self.removeSparseEntries(iteratedDF, threshold1, col1)
            print("iter=", i, ", rem1")
            print(len(iteratedDF))
            iteratedDF = self.removeSparseEntries(iteratedDF, threshold2, col2)
            print("iter=", i, ", rem2")
            print(len(iteratedDF))

        return iteratedDF