

import numpy as np
from neural_network.loss import *

import time

class NetworkModel:
	def __init__(self, model):
		self.network_model = model
		self.grad_update= 0

	def predict(self, x):
		return self._forward_propagation(x)

	def train(self, x_train, y_train, batch_size = 1, loss = mse, derivative_loss = derivative_mse, epochs = 10, learning_rate = 0.1, verbose = True):
		
		assert( len(x_train) == len(y_train) ), "Length differs between input and output arrays"
		assert( batch_size <= len(x_train) ), "Each mini-batch should be less or equal than the number of sample training"
		assert( batch_size >= 1 ), "Batch size should be greater or equal than 1"
			
		batches = np.ceil( (len(x_train) / batch_size ) ).astype('int')

		for epoch in range(epochs):

			ptr_initial = 0
			ptr_stop = batch_size

			for step in range(batches):

				x_batch = x_train[ptr_initial:ptr_stop]
				y_batch = y_train[ptr_initial:ptr_stop]

				start_epoch = time.time()								
				error = self._step_per_epoch(x_batch, y_batch, loss, derivative_loss, learning_rate)					
				end_epoch = time.time()	
				run_time = (end_epoch - start_epoch) * 1000

				ptr_initial += batch_size
				ptr_stop += batch_size

				if verbose:
					print(f'epoch: {epoch+1:4}/{epochs}')
					print(f'step: {step+1:4}/{batches} --> {run_time:.4f} ms/step ---> Loss: {error:.4f} ')
		
		print("--- End of Training ---")
		print("Gradient updates: ", self.grad_update)
		print("Loss: ", error)

	def _step_per_epoch(self, x_train, y_train, loss, derivative_loss, learning_rate):
		error = 0
		for x, y in zip(x_train, y_train):

			output = self._forward_propagation(x)
			error = loss(y, output)
			self._back_propagation(y, output, derivative_loss, learning_rate)
			
			self.grad_update += 1
			
		return error / len(x_train)		
									
				
	def _back_propagation(self, y, output, derivative_loss, learning_rate):
		gradient = derivative_loss(y, output)
		reversed_layers = self.network_model[::-1]

		for layer in reversed_layers:
			gradient = layer.backward(gradient, learning_rate)	
		
	def _forward_propagation(self, input):				
		output = input

		for layer in self.network_model:
			output = layer.forward(output)
		return output
		

