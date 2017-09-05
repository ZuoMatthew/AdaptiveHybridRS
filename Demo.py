from Evaluation import *
from AdaptiveHybridRS import *

filename = "./data/propertyDetails.csv"
userCol = 1
itemCol = 6
pRatios = [0.6, 0.2, 0.2]
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
simMetList = ["pearson"]

print("########################################################")
print("Shared items=", nSharedItems, ", Shared users=", nSharedUsers, ", iterations = ", iterations)
print("########################################################")


master = MasterRS(filename, chunkSize, itemCol, userCol, colNames, nSharedItems, nSharedUsers, iterations, colTypes)
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

recDictCBF = master.getCBFRecommenderDict(newTrainDF, kernelList[0], nRecom)
simMatrix = master.getCollaborativeSimMatrix(newTrainDF, simMetList[0])
recDictCF = master.getCFRecommenderDict(newTrainDF, simMatrix, nRecom)

demoUsers = userList[:10]
emptyRow = pd.DataFrame(index=['None'], columns=newTrainDF.columns)
recomNum = 10
for du in demoUsers:
    viewedItems = newTrainDF[newTrainDF[userCol] == du]
    cbfDict = recDictCBF[du][:recomNum]
    cfDict = recDictCF[du][:recomNum]

    cbfItemsList = [(newTrainDF[newTrainDF[itemCol] == i].iloc[0]).to_frame().T for i in cbfDict]
    cfItemsList = [(newTrainDF[newTrainDF[itemCol] == j].iloc[0]).to_frame().T for j in cfDict]

    cbfItems = pd.concat(cbfItemsList)
    cfItems = pd.concat(cfItemsList)

    demoItems = pd.concat([viewedItems, emptyRow, cbfItems, emptyRow, cfItems])
    demoItems.to_csv("./train/misc/" + du + ".csv", sep=',')
