import numpy as np

class OurNeuralNetwork:
    def __init__(self):
        weights = np.array([0,1])
        bias = 0

        self.h1 = Neuron(weights, bias)
        self.h2 = Neuron(weights, bias)
        self.o1 = Neuron(weights, bias)

    def feedforward(self, x):
        out_h1 = self.h1.feedforward(x)
        out_h2 = self.h1.feedforward(x)

        out_o1 = self.o1.feedforward(np.array([out_h1, out_h2]))


        return out_o1

network = OurNeuralNetwork()
x = np.array([2, 3])
print(network.feedforward(x))
