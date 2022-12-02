import numpy as np
import math
from Initializer import Initializer

class He(Initializer):
    def __init__(self):
        super().__init__(self.he_initialize)
    
    def he_initialize(self, n_inputs, n_neurons):
        # Implement He Initialization
        r = math.sqrt(6/n_inputs)
        w = np.random.uniform(-r,r, size=(n_inputs, n_neurons))
        b = np.random.uniform(-r, r, size=(1, n_neurons))

        return w, b