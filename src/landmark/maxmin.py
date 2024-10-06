import numpy as np
# import array
## NOTE: Looks like using less memory with pythran is preferred over numpy vector ops!
## This is def slowered than the simple loop
## Update non-landmark points with distance to nearest landmark
# c_dist = np.sum(np.power(np.abs(P[candidate_pts,:] - P[c_lm]), 2), axis=1)
# min_mask = (c_dist < lm_dist[candidate_pts])
# lm_dist[candidate_pts] = np.where(min_mask, c_dist, lm_dist[candidate_pts])
# closest_lm[candidate_pts] = np.where(min_mask, c_lm, closest_lm[candidate_pts])


# pythran export to_nat_2(int, int, int)
def to_nat_2(i: int, j: int, n: int) -> int:
	return (n * i - i * (i + 1) / 2 + j - i - 1) if i < j else (n * j - j * (j + 1) / 2 + i - j - 1)


# pythran export furthest_first_traversal_pdist(float[:] order(C), int, int, float, int)
def furthest_first_traversal_pdist(D: np.ndarray, n: int, k: int, eps: float, seed: int) -> tuple:
	## Outputs
	indices = []  # array('I')
	radii = [np.inf]  # array('d')
	pred = [seed]  # array('I') # predecessor array

	## Setup constants
	candidate_pts = np.concatenate([np.arange(seed), np.arange(seed + 1, n)])
	# candidate_pts = list(range(seed)) + list(range(seed + 1, n))

	# closest_lm = np.full(n, seed)
	closest_lm = seed * np.ones(n, dtype=np.int64)
	indices.append(seed)

	## Preallocate distance vector for landmarks; one for each point
	cover_radius = np.inf
	# lm_dist = np.full(n, cover_radius)
	lm_dist = cover_radius * np.ones(n)

	## Generate the landmarks
	stop_reached = False
	while not stop_reached:
		c_lm = indices[-1]  ## update current landmark

		## TODO: the above can probably be simplified
		# cand_dist = np.sum(np.power(np.abs(P[c_lm] - P[candidate_pts]), 2), axis=1)
		max_landmark_ind = -1
		max_landmark_dist = -np.inf
		for ii, idx in enumerate(candidate_pts):
			c_dist = D[to_nat_2(c_lm, idx, n)]
			if c_dist < lm_dist[idx]:
				lm_dist[idx] = c_dist  # update minimum landmark distance
				closest_lm[idx] = c_lm  # track which landmark achieved the minimum

			if lm_dist[idx] > max_landmark_dist:
				max_landmark_dist = lm_dist[idx]
				max_landmark_ind = ii

		## Of the remaining candidate points, find the one with the maximum landmark distance
		## NOTE: This is the 'max' part of the 'maxmin' method
		# max_landmark_ind = np.argmax(lm_dist[candidate_pts])
		max_landmark = candidate_pts[max_landmark_ind]

		## If the iterator is valid, we have a new landmark, otherwise we're finished
		cover_radius = max_landmark_dist  # lm_dist[max_landmark_ind]
		stop_reached = len(indices) >= k or cover_radius < eps
		if not stop_reached:
			indices.append(max_landmark)
			radii.append(cover_radius)
			pred.append(closest_lm[max_landmark])
			# candidate_pts.pop(max_landmark_ind)
			candidate_pts = np.delete(candidate_pts, max_landmark_ind)
		else:
			cover_radius = 0.0
			stop_reached = True
	return indices, radii, pred


# pythran export furthest_first_traversal_pc(float[:,:] order(C), int, float, int, float)
def furthest_first_traversal_pc(P: np.ndarray, k: int, eps: float, seed: int, p: float) -> tuple:
	## Outputs
	indices = []  # array('I')
	radii = [np.inf]  # array('d')
	pred = [seed]  # array('I') # predecessor array

	## Setup constants
	n = P.shape[0]
	candidate_pts = np.concatenate([np.arange(seed), np.arange(seed + 1, n)])
	# candidate_pts = list(range(seed)) + list(range(seed + 1, n))

	# closest_lm = np.full(n, seed)
	closest_lm = seed * np.ones(n, dtype=np.int64)
	indices.append(seed)

	## Preallocate distance vector for landmarks; one for each point
	cover_radius = np.inf
	# lm_dist = np.full(n, cover_radius)
	lm_dist = cover_radius * np.ones(n)

	## Generate the landmarks
	stop_reached = False
	while not stop_reached:
		c_lm = indices[-1]  ## update current landmark

		## TODO: the above can probably be simplified
		# cand_dist = np.sum(np.power(np.abs(P[c_lm] - P[candidate_pts]), 2), axis=1)
		max_landmark_ind = -1
		max_landmark_dist = -np.inf
		for ii, idx in enumerate(candidate_pts):
			c_dist = np.sum(np.power(np.abs(P[c_lm] - P[idx]), p))
			if c_dist < lm_dist[idx]:
				lm_dist[idx] = c_dist  # update minimum landmark distance
				closest_lm[idx] = c_lm  # track which landmark achieved the minimum

			if lm_dist[idx] > max_landmark_dist:
				max_landmark_dist = lm_dist[idx]
				max_landmark_ind = ii

		## Of the remaining candidate points, find the one with the maximum landmark distance
		## NOTE: This is the 'max' part of the 'maxmin' method
		# max_landmark_ind = np.argmax(lm_dist[candidate_pts])
		max_landmark = candidate_pts[max_landmark_ind]

		## If the iterator is valid, we have a new landmark, otherwise we're finished
		cover_radius = max_landmark_dist  # lm_dist[max_landmark_ind]
		stop_reached = len(indices) >= k or cover_radius < eps
		if not stop_reached:
			indices.append(max_landmark)
			radii.append(cover_radius)
			pred.append(closest_lm[max_landmark])
			# candidate_pts.pop(max_landmark_ind)
			candidate_pts = np.delete(candidate_pts, max_landmark_ind)
		else:
			cover_radius = 0.0
			stop_reached = True
	return indices, radii, pred
