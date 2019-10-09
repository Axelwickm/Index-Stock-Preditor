import csv
from random import sample
import numpy as np

import torch

TrainingSetPercentage = 0.85
LookBack = 8
LookForward = 8

# Read the data from the csv-file into variables
with open("PrecosAjustados.csv") as csvFile:
    reader = csv.reader(csvFile, delimiter=";")
    headers = next(reader)
    rows = [row for row in reader]
    data = list(zip(*rows[:-2]))


date = data[0]
IBOV = data[1]
stocks = list(map(lambda x: list(map(lambda y: float(y.replace(",", ".")), x)), data[2:]))
betas = list(map(lambda x: float(x.replace(",", ".")), rows[-1][2:]))

# Normalize the data


# Split into training set and testing set
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
        #print(index-stepBack, index+1)
        for j in range(index-stepBack, index+1):
            if j < 0:
                break
            stockValues[-1] += stocks[stockIndex][j] / (index+1 - (index-stepBack))
    return stockValues


