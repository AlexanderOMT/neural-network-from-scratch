
import numpy as np


# Make custom instruction ?

def mse(y_result, y_prediction):
	return np.mean(np.power(y_prediction - y_result, 2))
		
def derivative_mse(y_result, y_prediction):
	return  2 * (y_prediction - y_result) / np.size(y_result)



