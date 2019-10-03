import csv
from random import sample

import torch

TrainingSetPercentage = 0.9

# Read the data from the csv-file into variables
with open("PrecosAjustados.csv") as csvFile:
    reader = csv.reader(csvFile, delimiter=";")
    headers = next(reader)
    rows = [row for row in reader]
    data = list(zip(*rows[:-2]))

date = data[0]
IBOV = data[1]
stocks = list(map(lambda x: float(x.replace(",", ".")), data[2:][0]))
betas = list(map(lambda x: float(x.replace(",", ".")), rows[-1][2:]))

# Normalize the data


# Split into training set and testing set
trainingSet = sample(range(0, len(date)), int(len(date)*TrainingSetPercentage))
testingSet = list(filter(lambda x: x not in trainingSet, range(0, len(date))))

print(len(trainingSet))
print(len(testingSet))
