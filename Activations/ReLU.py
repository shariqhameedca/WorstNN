from Activations.activation import Activation
import numpy as np

class ReLU(Activation):
    def __init__(self):
        relu = lambda x: np.max((0, x))
        relu_drv = lambda x: 1 if x != 0 else 0
        super().__init__(relu, relu_drv)
