from Layers.BaseLayer import Layer
import numpy as np

class Activation(Layer):
    def __init__(self, activation, activation_drv):
        # Both are functions
        self.activation = activation
        self.activation_drv = activation_drv

    def forward(self, input):
        self.input = input
        return self.activation(self.input)

    def backward(self, output_gradient, learning_rate):
        return np.multiply(output_gradient, self.activation_drv(self.input))