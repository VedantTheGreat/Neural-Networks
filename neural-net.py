import numpy as np
def sigmoid(x):
    return 1/(1+np.exp(-x))
class NeuralNet:
    def __init__(self, weights, biases):
        self.weights = weights
        self.biases = biases
    def forward_process(self, input):
        dot = np.dot(self.weights, input) + self.biases
        return sigmoid(dot)

# Example neuron
weights = np.array([0,1])
biases = 4
n = NeuralNet(weights, biases)
print(n.forward_process(np.array([2,3])))
class OurNet(NeuralNet):
    def __init(self):
        weights = np.array([0,1])
        biases = 0
        self.h1 = NeuralNet(weights, biases)
        self.h2 = NeuralNet(weights, biases)
        self.o1 = NeuralNet(weights, biases)
    def feed(self, x):
        out_h1 = self.h1.forward_process(x)
        out_h2 = self.h2.forward_process(x)
        out_o1 = self.o1.forward_process(np.array([out_h1, out_h2]))
        return out_o1






