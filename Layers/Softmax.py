import numpy as np
from BaseLayer import Layer

class Softmax(Layer):
    def forward(self, input):
        tmp = np.exp(input - max(input))
        self.output = tmp / np.sum(tmp)
        return self.output

    def backward(self, output_gradient, learning_rate):
        n = np.size(self.output)
        return np.dot((np.identity(n) - self.output.T) * self.output, output_gradient)