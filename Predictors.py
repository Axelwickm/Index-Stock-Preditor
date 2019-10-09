import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("PyTorch running on "+str(device))

class Predictor:
    def save(self, filepath):
        return

    def train(self, input, output):
        return

    def predict(self, input):
        pass


class Baseline(Predictor):
    def __init__(self, intputCount, outputCount):
        self.inputCount = intputCount
        self.outputCount = outputCount

    def predict(self, input):
        # Predict using last known value
        return np.full(self.outputCount, np.mean(input))


class Encoder(Predictor):
    class Network(nn.Module):
        def __init__(self, intputCount, outputCount):
            super(Encoder.Network, self).__init__()
            self.fc1 = nn.Linear(intputCount, 50)
            self.relu1 = nn.ReLU()
            self.fc2 = nn.Linear(50, 35)
            self.relu2 = nn.ReLU()
            self.fc3 = nn.Linear(35, outputCount)
            self.sigmoid3 = nn.Sigmoid()

        def forward(self, x):
            out = self.fc1(x)
            out = self.relu1(out)
            out = self.fc2(out)
            out = self.relu2(out)
            out = self.fc3(out)
            out = self.sigmoid3(out)
            return out

    def __init__(self, intputCount, outputCount, LossFunction):
        self.net = Encoder.Network(intputCount, outputCount)
        self.net.to(device)

        self.criterion = LossFunction
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.00006)

    def save(self, filepath):
        torch.save(self.net.state_dict(), filepath)

    def train(self, input, output):
        input = torch.from_numpy(input).float()
        input = torch.autograd.Variable(input)
        input.to(device)

        output = torch.from_numpy(output).float()
        output = torch.autograd.Variable(output)
        output.to(device)

        self.optimizer.zero_grad()
        netOut = self.net(input)

        loss_size = self.criterion(netOut, output)
        loss_size.backward()
        self.optimizer.step()
        loss_size_detached = loss_size.item()

        return loss_size_detached

    def predict(self, input):
        input = torch.from_numpy(input).float()
        input.to(device)
        netOut = self.net(input)
        return netOut.detach().numpy()