import numpy as np
from numpy.typing import ArrayLike
from typing import Optional
from .predicates import is_distance_matrix, is_point_cloud
import _maxmin # native extension module

def _invert_comb2(N: int):
	"""Find n such that choose(n, 2) == N"""
	from math import sqrt, floor, ceil
	lb = sqrt(2*N)
	poss_n = np.arange(floor(lb), ceil(lb+2)+1)
	valid_n = N == (poss_n * (poss_n - 1)) // 2
	assert sum(valid_n) == 1, f"Invalid N = {N}; could not invert binomial coefficient"
	return poss_n[valid_n].item()

def landmarks(X: ArrayLike, k: Optional[int] = 15, eps: Optional[float] = -1.0, seed: int = 0, radii: bool = False, metric = "euclidean"):
	'''
	Computes landmarks points for a point set or set of distance using the 'maxmin' method. 

	Parameters:
		X = (n x d) matrix of *n* points in *d* dimensions, a distance matrix, or a set of pairwise distances
		k = (optional) number of landmarks requested. Defaults to 15. 
		eps = (optional) covering radius to stop finding landmarks at. Defaults to -1.0, using *k* instead.
		seed = index of the initial point to be the first landmark. Defaults to 0.
		radii = whether to return the insertion radii as well as the landmark indices. Defaults to False. 
		metric = metric distance to use. Ignored if *a* is a set of distances. See details.

	Details: 
		- The first radius is always the diameter of the point set, which can be expensive to compute for high dimensions, so by default "inf" is used as the first covering radius 
		- If the 'metric' parameter is "euclidean", the point set is used directly, otherwise the pairwise distances are computed first via 'dist'
		- If 'k' is specified an 'eps' is not (or it's -1.0), then the procedure stops when 'k' landmarks are found. The converse is true if k = 0 and eps > 0. 
			If both are specified, the both are used as stopping criteria for the procedure (whichever becomes true first).
		- Given a fixed seed, this procedure is deterministic. 

	Returns a pair (indices, radii) where:
		indices = the indices of the points defining the landmarks; the prefix of the *greedy permutation* (up to 'k' or 'eps')
		radii = the insertion radii whose values 'r' yield a cover of 'a' when balls of radius 'r' are places at the landmark points.

	The maxmin method yields a logarithmic approximation to the geometric set cover problem.

	'''
	X = np.asanyarray(X) if not isinstance(X, np.ndarray) else X
	k = 0 if k is None else int(k)
	eps = -1.0 if eps is None else float(eps)
	seed = int(seed)
	is_dist_m = False
	if X.ndim == 1 or (is_dist_m := is_distance_matrix(X)):
		n = X.shape[0] if is_dist_m else _invert_comb2(len(X))
		X = X[np.triu_indices(X.shape[0], k=1)] if is_dist_m else X
		indices, ins_radii = _maxmin.maxmin(X, eps, k, n, True, seed)
	elif metric == "euclidean" and is_point_cloud(X):
		n = X.shape[0]
		indices, ins_radii = _maxmin.maxmin(X, eps, k, n, False, seed)
		ins_radii = np.sqrt(ins_radii)
	else:
		raise ValueError("Unknown input type detected. Must be a matrix of points, a distance matrix, or a set of pairwise distances.")
	
	## Check packing/coverage properties is satsified
	is_monotone = np.all(np.diff(-np.array(ins_radii)) >= 0.0)
	assert is_monotone, "Invalid metric: non-monotonically decreasing radii found."

	return(indices, ins_radii) if radii else indices
