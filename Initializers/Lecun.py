import numpy as np
import math
from Initializer import Initializer

class Lecun(Initializer):
    def __init__(self):
        super().__init__(self.lecun_initialize)
    
    def lecun_initialize(self, n_inputs, n_neurons):
        # Implement Lecun Initialization
        r = math.sqrt(3/n_inputs)
        w = np.random.uniform(-r,r, size=(n_inputs, n_neurons))
        b = np.random.uniform(-r, r, size=(1, n_neurons))

        return w, b