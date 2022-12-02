import numpy as np
import math
from Initializer import Initializer

class Xavier(Initializer):
    def __init__(self):
        super().__init__(self.xavier_initialize)
    
    def xavier_initialize(self, n_inputs, n_neurons):
        # Implement Xavier Initialization
        fan_avg = np.mean([n_inputs, n_neurons])
        r = math.sqrt(3/fan_avg)
        w = np.random.uniform(-r,r, size=(n_inputs, n_neurons))
        b = np.random.uniform(-r, r, size=(1, n_neurons))

        return w, b