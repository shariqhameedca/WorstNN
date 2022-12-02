from Activations.activation import Activation
import numpy as np

class Tanh(Activation):
    def __init__(self):
        tanh = lambda x: np.tanh(x)
        tanh_drv = lambda x: 1 - tanh(x) ** 2
        super().__init__(tanh, tanh_drv)

    
        