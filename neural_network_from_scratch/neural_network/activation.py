

import numpy as np
from neural_network.layer import Layer

class Activation(Layer):
	def __init__(self, activation, derivative_activation):
		self.activation = activation
		self.derivative_activation = derivative_activation
		
	def forward(self, input):
		self.input = input
		return self.activation(self.input)
		
	def backward(self, output_gradient, learning_rate):
		return np.multiply(output_gradient, self.derivative_activation(self.input))

