from Evaluation import *
from AdaptiveHybridRS import *

filename = "./data/propertyDetails.csv"
userCol = 1
itemCol = 6
pRatios = [0.6, 0.2, 0.2]
nJump = 50
cfWeight = 0.5
cbfWeight = 0.5
colNames = [0,1,2,3,4,5,6,7]
testType = "test"

colCategorical = [0, 7]
colContinuous = [3, 4, 5]
colDiscrete = [2]
colTypes = [colCategorical, colContinuous, colDiscrete]
chunkSize = 1000000
nSharedItems = 30
nSharedUsers = 30
iterations = 12
saveFile = "./train/fullData6.csv"
kernelList = ["cosine"]

print("########################################################")
print("Shared items=", nSharedItems, ", Shared users=", nSharedUsers, ", iterations = ", iterations)
print("########################################################")

master = MasterRS(filename, chunkSize, itemCol, userCol, colNames, nSharedItems, nSharedUsers, iterations, colTypes)
# finalData = master.consolidateData(filename)
# finalData.to_csv(saveFile, sep=',')
finalData = pd.read_csv(saveFile, index_col=0)
finalData.columns = finalData.columns.astype(int)

nRecom = len(set(finalData[itemCol]))
nTest = len(set(finalData[userCol]))

print("===== Begin partition =====")
eval = Evaluation(finalData, itemCol, userCol, pRatios, testType)
trainDF = eval.dp.trainDF
validDF = eval.dp.validDF
testDF = eval.dp.testDF

newTrainDF, newTestDF = eval.dp.getNewTrainTestSet()

userList = master.getUserList(newTrainDF)
itemList = master.getItemList(newTrainDF)

nCols = len(newTrainDF.columns)
permList = np.random.permutation(userList)
recDict1 = master.getCBFRecommenderDict(newTrainDF, kernelList[0], nRecom)
pointsCBF1 = eval.computeROCPoints(userList, recDict1, nRecom, nTest, nJump)
contentCollabPts = [pointsCBF1]
contentCollabNames = kernelList

for i in range(len(contentCollabPts)):
    pts = contentCollabPts[i]
    truePts = pts[0]
    falsePts = pts[1]
    ptsDF = pd.DataFrame(np.vstack([truePts, falsePts]).T)
    ptsDF.to_csv("./train/" + kernelList[i] + "1.csv", sep=',')

contentCollabAucs = eval.calculateAUC(contentCollabPts)
eval.plotROCCurve(contentCollabPts, contentCollabNames, contentCollabAucs)