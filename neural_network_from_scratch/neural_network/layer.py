

class Layer:
	def __init__(self):
		self.input, self.output = None, None
		
	def forward(self, input):
		pass
		
	def backward(self, output_gradient, learning_rate):
		pass
