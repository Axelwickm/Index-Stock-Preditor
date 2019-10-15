from collections import defaultdict
import csv
import numpy as np

import torch

import Predictors
import Train

PredictorList = Train.PredictorList


def loadModels():
    print("Loading models")
    for predictor in PredictorList:
        predictor.load("./models/" + predictor.__class__.__name__ + ".pth")


def performanceForStock(IBOV, stocks, datapoints):
    loss_sizes = {}
    for predictor in PredictorList:
        averageLoss = 0
        for ind in datapoints:
            inputData = np.concatenate((
                Train.getHistory(stocks[stockID], ind, steps=Train.LookBack),
                Train.getHistory(IBOV, ind, steps=Train.LookBackIBOV)))
            outputData = Train.getFuture(stocks[stockID], ind, steps=Train.LookForward)

            result = predictor.predict(inputData)

            loss = Train.LossFunction(torch.tensor(result, requires_grad=True, dtype=torch.float),
                torch.tensor(outputData, requires_grad=True, dtype=torch.float)).detach().numpy()

            averageLoss += loss / len(datapoints)
        loss_sizes[predictor.__class__.__name__] = averageLoss

    return loss_sizes


def saveToCSV(data):
    print("Writing data to file")
    with open("./data/StockEvaluation.csv", "w") as f:
        writer = csv.DictWriter(f, list(data[0].keys()), delimiter=";", lineterminator='\n')
        writer.writeheader()
        for datum in data:
            writer.writerow(datum)


if __name__ == "__main__":
    loadModels()

    headers, date, IBOV, stocks, betas = Train.loadData()
    availableData = Train.availableData(date, stocks)
    trainingSet, testingSet = Train.splitData(availableData)

    # Split testingSet by stock
    stockDict = defaultdict(list)
    for ind in testingSet:
        stockDict[ind[0]].append(ind[1])

    stocksPredictionPerformances = []
    for stockID in range(len(stocks)):
        loss_sizes = performanceForStock(IBOV, stocks, stockDict[stockID])
        stocksPredictionPerformances.append(loss_sizes)
        print(headers[stockID+2]+" (row "+str(stockID+2)+"): "+str(loss_sizes))

    saveToCSV(stocksPredictionPerformances)

