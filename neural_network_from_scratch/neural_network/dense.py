

from neural_network.layer import Layer
import numpy as np

class Dense(Layer):
	def __init__(self, input_size, output_size):	

		self.weights = np.random.randn(output_size, input_size).astype(float)
		self.bias = np.random.randn(output_size, 1).astype(float)


	def forward(self, input):
		self.input = input
		return np.dot(self.weights, self.input) + self.bias
		
	def backward(self, output_gradient, learning_rate):

		weights_gradient = np.dot(output_gradient, self.input.T)
		input_gradient = np.dot(self.weights.T, output_gradient)
		
		self.weights -= weights_gradient * learning_rate
		self.bias -= output_gradient * learning_rate
		
		return input_gradient
	
	def get_weights(self):
		return self.weights, self.bias

