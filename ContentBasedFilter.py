import numpy as np
import pandas as pd
import os
from scipy import stats
from sklearn.neighbors import KernelDensity

class ContentBasedFilter:
    def __init__(self, dataframe, colItems, colUsers, colContinuous, colCategorical, colDiscrete, kernel):
        self.dataframe = dataframe
        self.colItems = colItems
        self.colUsers = colUsers
        self.colContinuous = colContinuous
        self.colCategorical = colCategorical
        self.colDiscrete = colDiscrete
        self.kernel = kernel
        self.colVar = [self.colUsers, self.colItems]
        self.preprocessedDF = self.preprocessDataframe()

    # Creates dummy variables for all categorical variables in the given dataframe df
    def createDummyFeatures(self, df):
        catColHeaders = [col for col in self.colCategorical]
        # df[self.colCategorical[-1]] = pd.to_numeric(df[self.colCategorical[-1]])
        dummyDF = pd.get_dummies(df, prefix=catColHeaders, columns=catColHeaders)
        return dummyDF


    # Standardizes all continuous variables in the given dataframe df by taking away the mean and dividing by the stdev
    def normaliseFeatures(self, df):
        # Do we need to limit the ranges before normalisation?
        df[self.colContinuous] = df[self.colContinuous].apply(pd.to_numeric)  # Changes cont columns to floating pt vals

        for col in df.columns:
            if col in self.colContinuous:
                featureMean = df[col].mean()
                featureStd = df[col].std()
                df[col] = (df[col] - featureMean) / featureStd
        return df

    # Perform preprocessing on the object's dataframe by combining the dummy and standardization methods
    def preprocessDataframe(self):
        dummyDF = self.createDummyFeatures(self.dataframe)
        preprocessedDF = self.normaliseFeatures(dummyDF)
        return preprocessedDF

    def userViewedItems(self, user):
        itemsViewed = self.preprocessedDF[self.preprocessedDF[self.colUsers] == user]
        return itemsViewed

    def zeroAndRepeatedCols(self, user):
        procColNames = self.preprocessedDF.columns
        itemsViewed = self.userViewedItems(user)
        nonVarCols = [col for col in self.preprocessedDF.columns if col not in self.colVar]

        zeroCols = (itemsViewed == 0).all()        # Bool if col contains zero values only
        zeroColNames = [procColNames[i] for i in range(len(procColNames)) if zeroCols.iloc[i]]   # Gets names of such cols
        nonZeroCols = [col for col in nonVarCols if col not in zeroColNames]
        nonZeroDetails = itemsViewed[nonZeroCols]

        uniqueItemsPerCol = nonZeroDetails.apply(pd.Series.nunique)
        repeatedColNames = list(uniqueItemsPerCol[uniqueItemsPerCol == 1].index)

        return zeroColNames, repeatedColNames   # Returns names of cols with elements that are all zero and all same


    # Perform KDE for continuous variables only
    def kernelDensityEstimate(self, user, cols):
        # print(user)
        itemsViewed = self.userViewedItems(user)
        continuousItemDetails = itemsViewed[cols]
        # transposeItems = continuousItemDetails.T
        # kde = stats.gaussian_kde(transposeItems)

        # kernel = stats.gaussian_kde(
        #     values, bw_method=bw / np.asarray(values).std(ddof=1))
        continuousItemMat = continuousItemDetails.as_matrix()
        kde = KernelDensity(kernel=self.kernel).fit(continuousItemMat)
        return kde

    # Performs estimate of the probability mass distribution for the discrete and categorical variables
    def discreteJointPMF(self, user):
        zeroColNames, repeatedColNames = self.zeroAndRepeatedCols(user)
        itemsViewed = self.userViewedItems(user)
        removedCols = self.colVar + self.colContinuous + zeroColNames + repeatedColNames
        keptCols = itemsViewed.columns.drop(removedCols)
        uniqueItemDetails = itemsViewed[keptCols]   # Removes all but useful discrete/categorical cols

        nItems = len(uniqueItemDetails)
        freqDictList = []
        for kc in keptCols:
            freqTable = np.round(uniqueItemDetails[kc].value_counts() / nItems, 5)
            freqDict = freqTable.to_dict()
            freqDictList.append(freqDict)

        return keptCols, freqDictList


    # Simplify df to remove all zero-only columns and filter by values in cols with a single repeated value
    def createFilteredItems(self, user):
        zeroColNames, repeatedColNames = self.zeroAndRepeatedCols(user)
        itemsViewed = self.userViewedItems(user)
        contRange = 0.1

        # For cols where user only views items with 1 val, filter df by this val (e.g. user only views 1-bed properties)
        filteredDF = self.preprocessedDF.drop(self.colUsers, 1)
        uniqueItemsDF = filteredDF.drop_duplicates()
        for rcn in repeatedColNames:
            rcnVal = itemsViewed[rcn].iloc[0]
            uniqueItemsDF = uniqueItemsDF[uniqueItemsDF[rcn] == rcnVal]

        contItemDetails = uniqueItemsDF[self.colContinuous]
        uniqueItemsPerCol = contItemDetails.apply(pd.Series.nunique)
        repeatedContCols = list(uniqueItemsPerCol[uniqueItemsPerCol == 1].index)

        for rcc in repeatedContCols:
            rccVal = itemsViewed[rcc].iloc[0]
            uniqueItemsDF = uniqueItemsDF[uniqueItemsDF[rcc] < rccVal * (1 + contRange)]
            uniqueItemsDF = uniqueItemsDF[uniqueItemsDF[rcc] > rccVal * (1 - contRange)]

        dropCols = zeroColNames + repeatedColNames + repeatedContCols
        filteredItems = uniqueItemsDF.drop(dropCols, 1)
        filteredItemList = filteredItems[self.colItems]

        return filteredItems, filteredItemList


    def computeItemDensities(self, user, filteredItems):
        cols = [fi for fi in filteredItems.columns if fi in self.colContinuous]
        kde = self.kernelDensityEstimate(user, cols)

        itemDensities = []
        for index, row in filteredItems.iterrows():
            continuousVals = [row[cc] for cc in self.colContinuous]
            rowDensity = np.exp(kde.score_samples([continuousVals]))
            # rowDensity = kde(list(continuousVals))
            roundedDensity = np.round(rowDensity[0], 10)
            itemDensities.append(roundedDensity)

        return itemDensities


    def computeItemPMFs(self, user, filteredItems):
        discreteCols, freqDictList = self.discreteJointPMF(user)

        itemPMFs = []
        for index, row in filteredItems.iterrows():
            discreteItemDetails = [row[dc] for dc in discreteCols]
            itemMarginalProbs = []

            for i in range(len(discreteItemDetails)):
                ival = discreteItemDetails[i]
                idict = freqDictList[i]
                if (ival not in list(idict.keys())):
                    itemMarginalProbs.append(0)
                else:
                    itemMarginalProbs.append(idict[ival])
            pmf = np.prod(itemMarginalProbs)
            itemPMFs.append(np.round(pmf, 5))

        return itemPMFs


    def combineDistributions(self, user):
        filteredItems, filteredItemList = self.createFilteredItems(user)
        itemDensities = self.computeItemDensities(user, filteredItems)
        itemPMFs = self.computeItemPMFs(user, filteredItems)
        itemsViewed = self.userViewedItems(user)[self.colItems]

        combinedDistribs = []
        for i in range(len(filteredItemList)):
            prob = itemDensities[i] * itemPMFs[i]
            if filteredItemList.iloc[i] in list(itemsViewed):
                prob = 0
            combinedDistribs.append(prob)

        return combinedDistribs

    def makeRecommendation(self, user, nRecommend):
        _, filteredItemList = self.createFilteredItems(user)
        combinedDistribs = self.combineDistributions(user)

        bestItemIndices = np.argsort(-np.array(combinedDistribs))
        recomItemIndices = bestItemIndices
        recomFilteredItems = [filteredItemList.iloc[i] for i in recomItemIndices]

        itemList = set(list(self.preprocessedDF[self.colItems]))
        filteredOutItems = [item for item in itemList if item not in recomFilteredItems]
        recomItems = (recomFilteredItems + filteredOutItems)[:nRecommend]

        return recomItems

    def createRecommenderDict(self, userList, nRecommend):
        recommenderDict = {}
        counts = 0
        for user in userList:
            recomItems = self.makeRecommendation(user, nRecommend)
            recommenderDict[user] = recomItems
            if counts % 50 == 0: print("CBF user: ", counts, "/", len(userList))
            counts += 1
        return recommenderDict
