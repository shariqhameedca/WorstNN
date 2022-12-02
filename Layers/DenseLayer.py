from BaseLayer import Layer
import numpy as np
from Initializers.Xavier import Xavier

class Dense(Layer):
    def __init__(self, input_size, output_size, initializer=Xavier()):
        self.input_size = input_size
        self.output_size = output_size
        # Initialize weights randomly
        self.weights, self.bias = initializer.initialize(self.output_size, self.input_size)

    # Forward Propogation
    def forward(self, input):
        self.input = input
        # calculating weights sum of inputs and bias
        return np.dot(self.weights, self.input) + self.bias

    # Backward Propogation
    def backward(self, output_gradient, learning_rate):
        # Calculating gradients
        weights_gradients = np.dot(output_gradient, self.input.T)
        bias_gradients = output_gradient
        # Updating parameters
        self.weights -= learning_rate * weights_gradients
        self.bias -= learning_rate * output_gradient
        # Returning input gradients to pass them to next (previous in architecture) layer
        return np.dot(self.weights.T, output_gradient)

