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
    predictions = {}
    actuals = []
    for predictor in PredictorList:
        averageLoss = 0
        pred = []
        actual = []
        for ind in datapoints:
            inputData = np.concatenate((
                Train.getHistory(stocks[stockID], ind, steps=Train.LookBack),
                Train.getHistory(IBOV, ind, steps=Train.LookBackIBOV)))
            outputData = Train.getFuture(stocks[stockID], ind, steps=Train.LookForward)

            result = predictor.predict(inputData)

            loss = Train.LossFunction(torch.tensor(result, requires_grad=True, dtype=torch.float),
                torch.tensor(outputData, requires_grad=True, dtype=torch.float)).detach().numpy()

            averageLoss += loss / len(datapoints)
            pred.append(sum(result)/len(result))
            actual.append(sum(outputData)/len(outputData))
        loss_sizes[predictor.__class__.__name__] = averageLoss
        predictions[predictor.__class__.__name__] = pred
        actuals = actual

    return loss_sizes, predictions, actuals


def saveToCSV(data):
    print("Writing data to file")
    with open("./data/StockEvaluation.csv", "w") as f:
        writer = csv.DictWriter(f, list(data[0].keys()), delimiter=";", lineterminator='\n')
        writer.writeheader()
        for datum in data:
            writer.writerow(datum)


def savePredToCSV(pred, actuals):
    print("Writing pred and actuals to file")
    with open("./data/StockPredictions.csv", 'w') as f:
        headers = [str(idx)+" "+k for idx, val in enumerate(pred) for k in (list(val.keys())+["actual"])]
        print(headers)
        writer = csv.DictWriter(f, headers, delimiter=";", lineterminator='\n')
        writer.writeheader()
        for time in range(len(actuals[0])):
            print(str(time)+" / "+str(len(actuals[0])))
            data = {}
            for idx in range(len(pred)):
                for k in list(pred[idx].keys()):
                    if len(pred[idx][k]) <= time:
                        break
                    data[str(idx)+" "+k] = pred[idx][k][time]
                else:
                    data[str(idx)+" actual"] = actuals[idx][time]

            writer.writerow(data)


if __name__ == "__main__":
    loadModels()

    headers, date, IBOV, stocks, betas = Train.loadData()
    availableData = Train.availableData(date, stocks)
    #trainingSet, testingSet = Train.splitData(availableData)

    # Split testingSet by stock
    stockDict = defaultdict(list)
    for ind in availableData:
        stockDict[ind[0]].append(ind[1])

    stocksPredictionPerformances = []
    stockPredictions = []
    stockActuals = []
    for stockID in range(len(stocks)):
        loss_sizes, predictions, actual = performanceForStock(IBOV, stocks, stockDict[stockID])
        stocksPredictionPerformances.append(loss_sizes)
        stockPredictions.append(predictions)
        stockActuals.append(actual)
        print(headers[stockID+2]+" (row "+str(stockID+2)+"): "+str(loss_sizes))

    saveToCSV(stocksPredictionPerformances)
    savePredToCSV(stockPredictions, stockActuals)

