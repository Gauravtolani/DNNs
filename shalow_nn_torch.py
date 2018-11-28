import torch
import torch.nn as nn
import torch.functional as F

X = torch.tensor(([2, 9], [1, 5], [3, 6]), dtype=torch.float)
y = torch.tensor(([92], [86], [89]), dtype=torch.float)

xPredicted = torch.tensor(([4, 8]), dtype=torch.float)

X = X / torch.max(X, 0)[0]  # maximum of X array
xPredicted = xPredicted / torch.max(xPredicted, 0)[0]  # maximum of xPredicted (our input data for the prediction)
y = y / 100  # max test score is 100


class neuralNetwork(object):
    def __init__(self):
        # parameters
        self.input_size = 2
        self.hidden_size = 3
        self.output_size = 1
        self.num_epochs = 5
        self.learning_rate = 0.001

        # weights
        self.W1 = torch.rand(self.inputSize, self.outputSize)
        self.W2 = torch.rand(self.hiddenSize, self.outputSize)

    def forward(self, X):
        self.z = torch.mm(X, self.W1)  # dot product of X (input) and first set of 3x2 weights
        self.z2 = torch.sigmoid(self.z)
        self.z3 = torch.mm(self.z2, self.W2)  # dot product of hidden layer (z2) and second set of 3x1 weights
        o = torch.sigmoid(self.z3)  # final activation function
        return o

