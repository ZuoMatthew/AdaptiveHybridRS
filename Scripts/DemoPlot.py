import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from sklearn import metrics
import os

filepath = "./train/misc/"
dirList = os.listdir(filepath)
nRecom = 30

for i in range(len(dirList)):
    f = dirList[i]
    filename = filepath + f
    resultsDF = pd.read_csv(filename, index_col=0)
    noneIndex = list(resultsDF.index).index('None')

    viewedItems = resultsDF.iloc[:noneIndex, :]
    cbfItems = resultsDF.iloc[noneIndex+1:noneIndex+11, :]
    cfItems = resultsDF.iloc[noneIndex+12:noneIndex+22, :]

    eView = viewedItems.iloc[:, 4]
    nView = viewedItems.iloc[:, 3]
    eCBF = cbfItems.iloc[:, 4]
    nCBF = cbfItems.iloc[:, 3]
    eCF = cfItems.iloc[:, 4]
    nCF = cfItems.iloc[:, 3]

    plt.plot(eView, nView, label="Viewed items", linestyle='None', marker="o", markersize=3)
    plt.plot(eCBF, nCBF, label="CBF recommendations", linestyle='None', marker="x", markersize=6)
    plt.plot(eCF, nCF, label="CF recommendations", linestyle='None', marker="+", markersize=7)
    plt.xlabel("Easting")
    plt.ylabel("Northing")
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    plt.legend()
    plt.title('Locations of Viewed and Recommended Items', y=1.03)
    print(filename)
    plt.show()