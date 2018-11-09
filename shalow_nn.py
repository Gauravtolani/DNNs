import numpy as np

X = np.array(([2, 9], [1, 5], [3, 6]), dtype=float)
y = np.array(([92], [86], [89]), dtype=float)

X = X / np.amax(X, axis=0)
y = y / 100


# input : 3,2 matrix
# ouput : 3,1 matrix

class Neural_Network:
    def __init__(self):
        self.input_size = 2
        self.output_size = 1
        self.hidden_size = 3

        self.W1 = np.random.randn(self.input_size, self.hidden_size)
        self.W2 = np.random.randn(self.hidden_size, self.output_size)

    def forward(self, X):
        self.z = np.dot(X, self.W1)
        self.z2 = self.sigmoid(self.z)
        self.z3 = np.dot(self.z2, self.W2)
        o = self.sigmoid(self.z3)
        return o

    def sigmoid(self, s):
        return 1/(1+np.exp(-s))


NN = Neural_Network()
o = NN.forward(X)

print("output predicted", str(o))
print("actual output", str(y))
