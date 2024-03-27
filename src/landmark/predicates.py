import numpy as np

def is_distance_matrix(x: np.ndarray) -> bool:
	''' Checks whether 'x' is a distance matrix, i.e. is square, symmetric, and that the diagonal is all 0. '''
	x = np.array(x, copy=False)
	is_square = x.ndim == 2	and (x.shape[0] == x.shape[1])
	return(False if not(is_square) else np.all(np.diag(x) == 0))

def is_point_cloud(x: np.ndarray) -> bool: 
	''' Checks whether 'x' is a 2-d array of points '''
	return(isinstance(x, np.ndarray) and x.ndim == 2)
