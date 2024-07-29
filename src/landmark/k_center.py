"""Metric K-center approximation functions."""

from math import ceil, floor, sqrt
from typing import Optional

import numpy as np
from numpy.typing import ArrayLike

import _maxmin  # isort:skip

def _invert_comb2(N: int):
	"""Find n such that choose(n, 2) == N."""
	lb = sqrt(2 * N)
	poss_n = np.arange(floor(lb), ceil(lb + 2) + 1)
	valid_n = N == (poss_n * (poss_n - 1)) // 2
	assert sum(valid_n) == 1, f"Invalid N = {N}; could not invert binomial coefficient"
	return poss_n[valid_n].item()


def is_distance_matrix(x: np.ndarray) -> bool:
	"""Checks whether 'x' is a distance matrix, i.e. is square, symmetric, and that the diagonal is all 0."""
	x = np.array(x, copy=False)
	is_square = x.ndim == 2 and (x.shape[0] == x.shape[1])
	return False if not (is_square) else np.all(np.diag(x) == 0)


def is_point_cloud(x: np.ndarray) -> bool:
	"""Checks whether 'x' is a 2-d array of points."""
	return isinstance(x, np.ndarray) and x.ndim == 2


P_MAP = {"euclidean": 2, "manhattan": 1, "cityblock": 1, "chebychev": np.inf, "minkowski": 2}


def landmarks(
	X: ArrayLike,
	k: Optional[int] = 15,
	eps: Optional[float] = -1.0,
	seed: int = 0,
	full_output: bool = False,
	metric: str = "euclidean",
	**kwargs: dict,
) -> np.ndarray:
	"""Computes *landmark* indices for a point set or metric space via the farthest-first traversal.
	
	This function computes a prefix of the *greedy permutation* of `X` using a greedy strategy known to \
	yield a 2-approximation for the *metric k-center* problem. 	For more details on the greedy permutation \
	and the metric k-center problem, see [1] and [2].

	Setting `k` constructs a fixed sized prefix, while setting `eps > 0` dynamically expands the prefix \
	until a cover over `X` of balls with radius `eps` is found. If `full_output = True`, a dictionary containing the \
	insertion radii and predecessors associated with each point in the prefix is returned. 
	
	Parameters:
		X: (n x d) matrix of *n* points in *d* dimensions, a distance matrix, or a set of pairwise distances
		k: number of landmarks requested. Defaults to 15. 
		eps: covering radius to stop finding landmarks at. If negative, uses *k* instead (default).
		seed: index of the initial point to be the first landmark. Defaults to 0.
		full_output: whether to return insertion radii and predecessors. Defaults to False. 
		metric: metric distance to use. Ignored if `X` is a set of distances. See details.
		**kwargs: If `metric = 'minkowski'`, supply `p` for the Minkowski p-norm.

	Returns:
		Indices of the landmark points; if `full_output = True`, also returns a dictionary containing auxiliary information.

	Notes:
		- By default `np.inf` is used as the first covering radius, as the diameter can be difficult to compute.
		- If the metric is a Minkowksi metric, the landmarks may be computed from the point set directly. For all \
			other metrics, you must supply all pairwise distances as the `X` argument. 
		- If both `k` and `eps` are specified, both are used as stopping criteria (whichever becomes true first).

	References:
		1. Eppstein, David, Sariel Har-Peled, and Anastasios Sidiropoulos. "Approximate greedy clustering and distance selection for graph metrics." arXiv preprint arXiv:1507.01555 (2015).
		2. Agarwal, Pankaj K., and Cecilia Magdalena Procopiuc. "Exact and approximation algorithms for clustering." Algorithmica 33 (2002): 201-226.
	"""
	X = np.asanyarray(X) if not isinstance(X, np.ndarray) else X
	k = 0 if k is None else int(k)
	eps = -1.0 if eps is None else float(eps)
	seed = int(seed)
	is_dist_m = False
	if X.ndim == 1 or (is_dist_m := is_distance_matrix(X)):
		n = X.shape[0] if is_dist_m else _invert_comb2(len(X))
		X = X[np.triu_indices(X.shape[0], k=1)] if is_dist_m else X
		indices, ins_radii, pred_map = _maxmin.maxmin(X, eps, k, n, True, seed, 2)
	elif is_point_cloud(X):
		if isinstance(metric, str):
			assert metric.lower() in P_MAP, f"Unknown metric '{metric}'; must be one of {str(list(P_MAP.keys()))}"
			p = kwargs.pop("p", P_MAP[metric.lower()])
			p = -1 if np.isinf(p) else np.abs(float(p))
		else:
			raise ValueError(f"Invalid input metric '{metric}'; Must be a known minkowski metric.")
		n = X.shape[0]
		indices, ins_radii, pred_map = _maxmin.maxmin(X, eps, k, n, False, seed, p)
		ins_radii = np.sqrt(ins_radii)
	else:
		raise ValueError(
			"Unknown input type detected. Must be a matrix of points, a distance matrix, or a set of pairwise distances."
		)

	## Check packing/coverage properties is satisfied
	is_monotone = np.all(np.diff(-np.array(ins_radii)) >= 0.0)
	assert is_monotone, "Invalid metric: non-monotonically decreasing radii found."

	## Switch the input
	if full_output:
		return (indices, {"radii": ins_radii, "predecessors": pred_map})
	else:
		return indices
