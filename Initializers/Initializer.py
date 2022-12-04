import numpy as np
import math

class Initializer:
    def __init__(self, technique):
        self.technique = technique

    def initialize(self, n_neurons, n_inputs):
        self.n_neurons = n_neurons
        self.n_inputs = n_inputs

        self.weights, self.bias = self.technique(self.n_neurons, self.n_inputs)
        return self.weights, self.bias




