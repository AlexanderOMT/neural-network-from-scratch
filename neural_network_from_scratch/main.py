

from neural_network.dense import Dense
from neural_network.loss import *
from neural_network.activation_function import *
from neural_network.model import NetworkModel


import numpy as np


f_x = lambda x: 1.8 * x + 32
x = np.array([-40, -10,  0,  8, 15, 22,  38, 10, -5, -50, 5, 28],  dtype=float)
y = np.array([-40,  14, 32, 46, 59, 72, 100, 50, 23, -58, 41, 82.4],  dtype=float)

network = [
    Dense(1, 1),
    Dense(1, 2),
    ReLU(), 
    Dense(2, 2),
    Dense(2, 1),
    Dense(1, 1)
]

model = NetworkModel(network)

model.train(x, y, learning_rate = 0.00001, batch_size = len(x)//1, epochs = 1000, verbose=True)

x_pred = 16
y_result = f_x(x_pred)

y_pred = model.predict([x_pred])

print(f'Prediction : {y_pred} vs Real Result : {y_result} => Loss for Prediction : {mse(y_result, y_pred)} ')

