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
coldRatio = 0.95
saveFile = "./train/fullData6.csv"
kernelList = ["cosine"]
simMetList = ["pearson"]

print("##########################################")
print("Shared items=", nSharedItems, ", Shared users=", nSharedUsers)
print("##########################################")
print("Cold ratio = ", str(coldRatio))

master = MasterRS(filename, chunkSize, itemCol, userCol, colNames, nSharedItems, nSharedUsers, iterations, colTypes)
# finalData = master.consolidateData(filename)
# finalData.to_csv(saveFile, sep=',')
finalData = pd.read_csv(saveFile, index_col=0)
finalData.columns = finalData.columns.astype(int)

print("===== Begin partition =====")
eval = Evaluation(finalData, itemCol, userCol, pRatios, testType)
trainDF = eval.dp.trainDF
validDF = eval.dp.validDF
testDF = eval.dp.testDF

newTrainDF1, newTestDF1 = eval.dp.getNewTrainTestSet()
newTrainDF, newTestDF = eval.dp.coldPartition(coldRatio)

userList1 = master.getUserList(newTrainDF1)
itemList1 = master.getItemList(newTrainDF1)

userList = master.getUserList(newTrainDF)
itemList = master.getItemList(newTrainDF)
nRecom = len(itemList)
nTest = len(userList)

nCols = len(newTrainDF.columns)
permList = np.random.permutation(userList)

simMatrix = master.getCollaborativeSimMatrix(newTrainDF, simMetList[0])
recDictCF1 = master.getCFRecommenderDict(newTrainDF, simMatrix, nRecom)
pointsCF1 = eval.computeROCPoints(userList, recDictCF1, nRecom, nTest, nJump)

contentCollabPts = [pointsCF1]
contentCollabNames = ["Collaborative Filtering"]

for i in range(len(contentCollabPts)):
    pts = contentCollabPts[i]
    truePts = pts[0]
    falsePts = pts[1]
    ptsDF = pd.DataFrame(np.vstack([truePts, falsePts]).T)
    ptsDF.to_csv("./train/cold" + str(coldRatio) + ".csv", sep=',')

contentCollabAucs = eval.calculateAUC(contentCollabPts)
eval.plotROCCurve(contentCollabPts, contentCollabNames, contentCollabAucs)
