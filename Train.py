import csv
from random import sample

import numpy as np
from visdom import Visdom

import torch
import torch.nn as nn

import Predictors


LookBack = 8
LookForward = 8

TrainingSetPercentage = 0.85
TrainingEpochs = 30
AverageEvery = 500
ShouldPlot = True
LossFunction = nn.SmoothL1Loss()

PredictorList = [
    Predictors.Baseline(LookBack, LookForward,),
    Predictors.Encoder(LookBack, LookForward, LossFunction)
]

# Read the data from the csv-file into variables
print("Loading data")
with open("PrecosAjustados.csv") as csvFile:
    reader = csv.reader(csvFile, delimiter=";")
    headers = next(reader)
    rows = [row for row in reader]
    data = list(zip(*rows[:-2]))


date = data[0]
IBOV = np.array(data[1])
stocks = np.array(list(map(lambda x: list(map(lambda y: float(y.replace(",", ".")), x)), data[2:])))
betas = np.array(list(map(lambda x: float(x.replace(",", ".")), rows[-1][2:])))

# Normalize the data
stocks /= np.max(stocks)

# Split into training set and testing set
print("Splitting into training and testing set")
availableData = np.array(list(np.ndindex((len(stocks), len(date)-LookBack-LookForward))))
availableData[:, 1] += LookBack
availableData = list(map(tuple, availableData))
trainingSet = sample(availableData, int(len(availableData)*TrainingSetPercentage))
testingSet = list(filter(lambda x: x not in trainingSet, availableData))


def getHistory(stockIndex, index, steps=5):
    stockValues = []
    for i in range(steps):
        stepBack = int(pow(i+1, 1.5))
        stockValues.append(0)
        minIndex = max(0, index-stepBack)
        for j in range(minIndex, index+1):
            if j < 0:
                break
            stockValues[-1] += stocks[stockIndex][j] / (index+1 - minIndex)
    return np.array(stockValues, dtype=np.float)


def getFuture(stockIndex, index, steps=5):
    stockValues = []
    for i in range(steps):
        stepForward = int(pow(i+1, 1.5))
        stockValues.append(0)
        maxIndex = min(len(stocks[stockIndex])-1, index+stepForward+1)
        for j in range(index+1, maxIndex):
            if len(stocks[stockIndex]) <= j:
                break
            stockValues[-1] += stocks[stockIndex][j] / (maxIndex - (index+1))
    return np.array(stockValues, dtype=np.float)


#  Train the predictors
print("Instantiating predictors")
if ShouldPlot:
    vis = Visdom()
    loss_window = vis.line(X=np.zeros((1)),
                           Y=np.zeros((1)),
                           )

for predictor in PredictorList:
    print("Training: "+predictor.__class__.__name__+" predictor")
    performanceLog = []

    for epoch in range(TrainingEpochs):
        print("Epoch "+str(epoch)+":")
        epochAverage = 0
        smallAverage = 0

        for i in range(len(trainingSet)):
            indices = trainingSet[i]
            inputData = getHistory(indices[0], indices[1], steps=LookBack)
            outputData = getHistory(indices[0], indices[1], steps=LookForward)
            loss_size = predictor.train(inputData, outputData)
            if loss_size is None:
                print("This predictor cannot be trained\n")
                break

            epochAverage += loss_size / len(trainingSet)
            smallAverage += loss_size / AverageEvery

            if (i+1) % AverageEvery == 0:
                performanceLog.append(smallAverage)
                smallAverage = 0

                if ShouldPlot:
                    vis.line(
                        X=np.linspace(0, len(performanceLog)*AverageEvery, len(performanceLog)),
                        Y=np.array(performanceLog),
                        win = loss_window,
                        name=predictor.__class__.__name__,
                        opts=dict(
                            xlabel='training steps',
                            ylabel='loss size',
                            ytype='log'
                        )
                    )
        else:
            print(str(epochAverage)+"\n")
            continue
        break

# Validate the predictors
for predictor in PredictorList:
    print("Validating: " + predictor.__class__.__name__ + " predictor")
    averageLoss = 0
    for i in range(len(testingSet)):
        indices = testingSet[i]
        inputData = getHistory(indices[0], indices[1], steps=LookBack)
        outputData = getHistory(indices[0], indices[1], steps=LookForward)
        result = predictor.predict(inputData)
        loss = LossFunction(torch.tensor(result, requires_grad=True, dtype=torch.float),
                            torch.tensor(outputData, requires_grad=True, dtype=torch.float)).detach().numpy()
        averageLoss += loss / len(testingSet)
    print("Average "+LossFunction.__class__.__name__+" performance: "+str(averageLoss)+"\n")

