import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("PyTorch running on "+str(device))

class Predictor:
    def save(self, filepath):
        return

    def load(self, filepath):
        return

    def train(self, input, output):
        return

    def predict(self, input):
        pass


class Baseline(Predictor):
    def __init__(self, inputCount, outputCount):
        self.inputCount = inputCount
        self.outputCount = outputCount

    def predict(self, input):
        # Predict using last known value
        return np.full(self.outputCount, np.mean(input))


class FIR(Predictor):
    trainEpoch = True

    class Filter(nn.Module):
        def __init__(self, inputCount, outputCount):
            super(FIR.Filter, self).__init__()
            self.weights = nn.Parameter(torch.ones(inputCount, requires_grad=True))
            torch.nn.init.uniform_(self.weights, -1, 1)
            self.outputCount = outputCount

            self.seq = nn.Sequential()

        def predict(self, x):
            output = torch.autograd.Variable(torch.Tensor(self.outputCount))
            for i in range(self.outputCount):
                output[i] = torch.sum(self.weights * x)
                x = torch.cat((x[1:], torch.tensor([output[i]])))

            return output

    def __init__(self, inputCount, outputCount, LossFunction):
        self.inputCount = inputCount
        self.outputCount = outputCount

        self.filter = FIR.Filter(inputCount, outputCount)

        self.criterion = LossFunction
        self.optimizer = torch.optim.Adam(self.filter.parameters(), lr=0.0001)

    def save(self, filepath):
        torch.save(self.filter.state_dict(), filepath)

    def load(self, filepath):
        self.filter.load_state_dict(torch.load(filepath))
        self.filter.eval()

    def train(self, input, output):
        input = torch.from_numpy(input).float()
        input = torch.autograd.Variable(input)
        input.to(device)

        output = torch.from_numpy(output).float()
        output = torch.autograd.Variable(output)
        output.to(device)

        self.optimizer.zero_grad()
        filterOut = self.filter.predict(input)

        loss_size = self.criterion(filterOut, output)
        self.optimizer.zero_grad()
        loss_size.backward()

        self.optimizer.step()
        loss_size_detached = loss_size.item()

        return loss_size_detached

    def predict(self, input):
        pass



class FC_Net(Predictor):
    class Network(nn.Module):
        def __init__(self, inputCount, outputCount):
            super(FC_Net.Network, self).__init__()
            self.fc1 = nn.Linear(inputCount, 50)
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

    def __init__(self, inputCount, outputCount, LossFunction):
        self.net = FC_Net.Network(inputCount, outputCount)
        self.net.to(device)

        self.criterion = LossFunction
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.00006)

    def save(self, filepath):
        torch.save(self.net.state_dict(), filepath)

    def load(self, filepath):
        self.net.load_state_dict(torch.load(filepath))
        self.net.eval()

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
