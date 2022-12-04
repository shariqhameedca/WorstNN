import numpy as np

from Layers.DenseLayer import Dense
from Activations.Tanh import Tanh
from Losses.Loss import mse, mse_drv
from Initializers.Xavier import Xavier
from model import train, test, predict

X = np.reshape([[0,0],[0,1],[1,0],[1,1]], (4,2,1))
y = np.reshape([[0],[1],[1],[0]], (4,1,1))

xavier = Xavier()
model = [
    Dense(2, 3, initializer=xavier),
    Tanh(),
    Dense(3, 1, initializer=xavier),
    Tanh()
]

# train
train(model, mse, mse_drv, X, y, epochs=100, lr=0.1)

# Trying random values with our model
x_ = np.linspace(0,1,20)
y_ = np.linspace(0,1,20)

for x, y in zip(x_,y_):
    prediction = predict(model, [[x], [y]])
    print(prediction)