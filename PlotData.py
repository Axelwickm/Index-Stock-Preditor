import csv
import numpy as np
import matplotlib.pyplot as plt


def loadPredictions():
    stocks = {}
    with open("./data/StockPredictions.csv", 'r') as f:
        reader = csv.reader(f, delimiter=";")
        headers = next(reader)
        for i in range(0, len(headers), 4):
            id = headers[i].split()[0]
            stocks[id] = {}
            stocks[id]["baseline"] = []
            stocks[id]["FIR"] = []
            stocks[id]["FC_Net"] = []
            stocks[id]["actual"] = []

        data = [row for row in reader]
        for row in data:
            for i in range(0, len(row), 4):
                id = headers[i].split()[0]
                stocks[id]["baseline"].append(float(row[i]))
                stocks[id]["FIR"].append(float(row[i+1]))
                stocks[id]["FC_Net"].append(float(row[i+2]))
                stocks[id]["actual"].append(float(row[i+3]))
    return stocks


def plotPredictions(data, stockToPlot):
    print("Plotting stock #"+stockToPlot)
    data = data[stockToPlot]
    fig = plt.figure(figsize=(8, 8))
    fig.suptitle("Stock #"+stockToPlot, fontsize=16)
    predictors = ["baseline", "FIR", "FC_Net"]
    predictorColors = {"baseline": "#70db84", "FIR": "#e3ca78", "FC_Net": "#56bde3"}
    for pred in predictors:
        if pred == "baseline":
            ax = fig.add_subplot(2, 2, 1)
        elif pred == "FIR":
            ax = fig.add_subplot(2, 2, 2)
        elif pred == "FC_Net":
            ax = fig.add_subplot(2, 1, 2)
        ax.set_title(pred)

        for i in range(len(data[pred]) - 1):
            plt.plot([i, i + 2], [data["actual"][i], data[pred][i]], color=predictorColors[pred])

        ax.plot(data["actual"], color="black")


    #plt.show()
    plt.savefig("./plots/predictions_"+stockToPlot+".png")


if __name__ == "__main__":
    stockPredictions = loadPredictions()
    for i in range(len(stockPredictions)):
        plotPredictions(stockPredictions, str(i))
