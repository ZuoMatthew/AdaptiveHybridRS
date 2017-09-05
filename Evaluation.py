import pandas as pd
import numpy as np
import random
from sklearn import metrics
import matplotlib.pyplot as plt
from MasterRS import *
from CollaborativeFilter import *
from DataPartition import *

class Evaluation:
    def __init__(self, consolidatedDF, colItems, colUsers, partitionRatios, testOrValid):
        self.colItems = colItems
        self.colUsers = colUsers
        self.consolidatedDF = consolidatedDF
        self.testOrValid = testOrValid
        self.dp = DataPartition(consolidatedDF, colItems, colUsers, partitionRatios, testOrValid)

    def computePositiveRates(self, user, recommendedList):
        # itemsViewed = self.userItemDict[user]
        newTrainDF, newTestDF = self.dp.getNewTrainTestSet()
        allTrainItems = self.dp.getAllTrainItems(newTrainDF)

        testItems = self.dp.getTestItems(user, newTrainDF)

        nUserTest = len(testItems)
        if nUserTest == 0:
            tpRate = -1
            fpRate = -1
        else:
            truePosItems = [item for item in testItems if item in recommendedList]
            falseNegItems = list(set(testItems) - set(truePosItems))
            trueNegItems = list((set(allTrainItems) - set(recommendedList)) - set(testItems))
            falsePosItems = list(set(recommendedList) - set(truePosItems))

            try:
                tpRate = len(truePosItems)/(len(truePosItems) + len(falseNegItems))     # tpr = tp/(tp+fn)
            except ZeroDivisionError:
                tpRate = 0
            try:
                fpRate = len(falsePosItems)/(len(falsePosItems) + len(trueNegItems))    # fpr = fp/(fp+tn)
            except ZeroDivisionError:
                fpRate = 0

        return tpRate, fpRate


    def computeROCPoints(self, userList, recommendationDict, nRecom, nTest, nJump):
        truePosMeans = []
        falsePosMeans = []
        permutatedUserList = np.random.permutation(userList)
        print("ROC")
        print(len(permutatedUserList))
        print(nTest)
        for iRec in np.arange(0, nRecom, nJump):
            truePosPoints = []
            falsePosPoints = []

            for u in range(0, nTest-1):
                user = permutatedUserList[u]
                recomItems = recommendationDict[user][:iRec]
                iTruePos, iFalsePos = self.computePositiveRates(user, recomItems)
                if iTruePos >= 0 and iFalsePos >= 0:
                    truePosPoints.append(iTruePos)
                    falsePosPoints.append(iFalsePos)

            truePosMeans.append(np.mean(truePosPoints))
            falsePosMeans.append(np.mean(falsePosPoints))

            print("Progress: ", iRec, "/", nRecom)
        return [falsePosMeans, truePosMeans]


    def calculateAUC(self, algoPointsArr):
        algoAUCArr = []
        for algo in algoPointsArr:
            falsePosPoints = algo[0]
            truePosPoints = algo[1]
            auc = metrics.auc(falsePosPoints, truePosPoints)
            algoAUCArr.append(auc)

        return algoAUCArr


    # tpfp is a 3D array, consisting of an array of [[falsePos], [truePos]] points for each algorithm to be plotted
    def plotROCCurve(self, algoPointsArr, algoNamesArr, algoAUCArr):
        nAlgos = len(algoPointsArr)
        for i in range(nAlgos):
            algoName = algoNamesArr[i]
            ftpa = algoPointsArr[i]

            falsePosPts = ftpa[0]   # Array of averaged false positive points
            truePosPts = ftpa[1]    # Array of averaged true positive points
            aucVal = np.round(algoAUCArr[i], 3)
            label = algoName + " (Area = " + str(aucVal) + ")"
            plt.plot(falsePosPts, truePosPts, linewidth=1.5, label=label)

        xyLine = np.arange(0, 1.1, 0.5)
        plt.plot(xyLine, xyLine, linestyle="dashed", color="k")
        plt.xlim(0.0, 1.01)
        plt.ylim(0.0, 1.01)
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")
        plt.title('ROC Curve of Recommender Systems')

        nLegendCols = np.min([3, len(algoNamesArr)])
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=nLegendCols)
        plt.show()