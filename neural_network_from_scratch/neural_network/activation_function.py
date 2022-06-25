

from neural_network.activation import Activation
import numpy as np

class Tanh(Activation):

	def __init__(self):
	
		tahn = lambda x: np.tanh(x)		
		derivative_tahn = lambda x: 1. - np.tanh(x) ** 2		
		super().__init__(tahn, derivative_tahn)
 
 
class ReLU(Activation):

	def __init__(self):
	
		relu = lambda x: np.maximum(0, x)		
		derivative_leaky_relu = lambda x: ( 1. * (x >= 0) ) + ( 0.0001 * x * (x < 0) )	
		derivative_relu = lambda x: np.array(x > 0).astype('int')	
		super().__init__(relu, derivative_leaky_relu)       	

