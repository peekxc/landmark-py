# %% 
import cv2
import numpy as np 
from landmark import landmarks
from scipy.spatial.distance import cdist, pdist, squareform
from bokeh.plotting import figure, show
from bokeh.models import SetValue, Slider, CustomJS
from bokeh.layouts import column

# %% 
## Load image with OpenCV
image = cv2.imread('/Users/mpiekenbrock/landmark-py/notebooks/parrots/parrot.jpeg', flags=cv2.IMREAD_GRAYSCALE)

## Get pixel landmarks + representatives
pixel_int = np.ravel(image)[:,np.newaxis]
ind = landmarks(pixel_int, k=50)

# ## Write compressed images
# for k in np.arange(2,31):
#   pixel_rep = pixel_int[ind[:k]]
#   pixel_lms = cdist(pixel_rep, pixel_int).argmin(axis=0)
#   img_compressed = pixel_rep[pixel_lms].reshape(image.shape)
#   cv2.imwrite(f'/Users/mpiekenbrock/landmark-py/notebooks/parrots/parrot_{k}.jpeg', img_compressed)

# %% 
base_path = '/Users/mpiekenbrock/landmark-py/notebooks'
w,h = image.shape
p = figure(height=w, width=h, x_range=(0,w), y_range=(0,h))

img_g = p.image_url(url=[base_path + '/parrots/parrot_5.jpeg'], x=w/2, y=h/2, w=w, h=h, anchor="center")
p.xaxis.visible = False
p.yaxis.visible = False
p.toolbar_location = None
p.image_rgba()
# show(p)

# %% Slider
# callback = SetValue(obj=img_g, attr="label", value="Bar")
k_slider = Slider(start=2, end=10, value=2, step=1, title="Number of Landmarks")
# k_slider.js_link('value', img_g.data_source, 'url')
cg_callback = CustomJS(args=dict(source=img_g.data_source), code="""
  const k = cb_obj.value
  console.log(source.data)
  source.data = {
    url: ['/Users/mpiekenbrock/landmark-py/notebooks/parrots/parrot_' + k + '.jpeg']
  }
""")
k_slider.js_on_change('value', cg_callback)
q = column(k_slider, p)
show(q)

# wut = np.c_[pixel_rep[pixel_lms],pixel_rep[pixel_lms],pixel_rep[pixel_lms],255*np.ones(len(pixel_lms)).astype(np.uint8)].reshape(image.shape + (4,))
# p.image_rgba([wut.view(np.uint32)], x=w/2, y=h/2, dw=w, dh=h, anchor="center")
# show(p)

from bokeh.io import output_notebook
output_notebook(hide_banner=True)


from set_cover.covers import cmds
cmds(pdist(pixel_int)**2)



# d = 2
# subset = ind
# D, n = squareform(pdist(pixel_int[subset], metric="euclidean")**2), len(pixel_int)
# S = cdist(pixel_int[subset], pixel_int, metric="euclidean")**2
	
# ## At this point, D == distance matrix of landmarks points, S == (J x n) distances to landmarks
# evals, evecs = cmds(D, d=2, coords=False)

# ## Interpolate the lower-dimension points using the landmarks
# mean_landmark = np.mean(D, axis = 1).reshape((D.shape[0],1))
# w = np.where(evals > 0)[0]
# L_pseudo = evecs/np.sqrt(evals[w])
# Y = np.zeros(shape=(n, d))
# Y[:,w] = (-0.5*(L_pseudo.T @ (S.T - mean_landmark.T).T)).T 

# Y

# def landmark_mds(X: ArrayLike, d: int = 2, L: Union[ArrayLike, int, str] = "default", normalize=False, ratio=1.0, prob=1.0):
# 	''' 
# 	Landmark Multi-Dimensional Scaling 
	
# 	Parameters: 
# 		X := point cloud matrix, distance matrix, set of pairwise distances.
# 		d := target dimension for the coordinatization
# 		L := either an integer specifying the number of landmarks to use, or indices of 'X' designating which landmarks to use
# 		normalize := whether to re-orient the embedding using PCA to reflect the distribution of 'X' rather than distribution of landmarks. Defaults to false.   
# 		ratio := aspect ratio between the smallest/largest dimensions of the bounding box containing 'X'. Defaults to 1. See details.
# 		prob := probability the embedding should match exactly with the results of MDS. See details. 

# 	Details: 
# 		This function uses landmark	points and trilateration to compute an approximation to the embedding obtained by classical 
# 		multidimensional scaling, using the technique described in [1].

# 		The parameter 'L' can be either an array of indices of the rows of 'X' indicating which landmarks to use, a single integer specifying the 
# 		number of landmarks to compute using maxmin, or "default" in which case the number of landmarks to use is calculated automatically. In the 
# 		latter case, 'ratio' and 'prob' are used to calculate the number of landmarks needed to recover the same embedding one would obtain using 
# 		classical MDS on the full (squared) euclidean distance matrix of 'X'. The bound is from [2], which sets the number of landmarks 'L' to: 

# 		L = floor(9*(ratio**2)*log(2*(d+1)/prob))

# 		Since this bound was anlyzed with respect to uniformly random samples, it tends to overestimate the number of landmarks needed compared to 
# 		using the maxmin approach, which is much more stable. In general, a good rule of thumb is choose L as some relatively small multiple of the 
# 		target dimension d, i.e. something like L = 15*d.

# 	References: 
# 		1. De Silva, Vin, and Joshua B. Tenenbaum. Sparse multidimensional scaling using landmark points. Vol. 120. technical report, Stanford University, 2004.
# 		2. Arias-Castro, Ery, Adel Javanmard, and Bruno Pelletier. "Perturbation bounds for procrustes, classical scaling, and trilateration, with applications to manifold learning." Journal of machine learning research 21 (2020): 15-1.

# 	'''
# 	if isinstance(L, str) and (L == "default"):
# 		L = int(9*(ratio**2)*np.log(2*(d+1)/prob))
# 		subset, _ = landmarks(X, k=L)
# 	elif isinstance(L, numbers.Integral):
# 		subset, _ = landmarks(X, k=L)
# 	else: 
# 		assert isinstance(L, np.ndarray)
# 		subset = L

# 	## Apply classical MDS to landmark points
# 	from itertools import combinations
# 	J = len(subset) 
# 	if is_pairwise_distances(X):
# 		D, n = as_dist_matrix(subset_dist(X, subset)), inverse_choose(len(X), 2)
# 		S = np.zeros(shape=(J,n))
# 		for j, index in enumerate(subset):
# 			for i in range(n):
# 				S[j,i] = 0.0 if i == index else X[rank_comb2(i,index,n)]
# 	elif is_distance_matrix(X):
# 		D, n = subset_dist(X, subset), X.shape[0]
# 		S = X[np.ix_(subset, range(n))]
# 	else:
# 		D, n = dist(X[subset,:], as_matrix=True, metric="euclidean")**2, X.shape[0]
# 		S = dist(X[subset,:], X, metric="euclidean")**2
	
# 	## At this point, D == distance matrix of landmarks points, S == (J x n) distances to landmarks
# 	evals, evecs = cmds(D, d=d, coords=False)

# 	## Interpolate the lower-dimension points using the landmarks
# 	mean_landmark = np.mean(D, axis = 1).reshape((D.shape[0],1))
# 	w = np.where(evals > 0)[0]
# 	L_pseudo = evecs/np.sqrt(evals[w])
# 	Y = np.zeros(shape=(n, d))
# 	Y[:,w] = (-0.5*(L_pseudo.T @ (S.T - mean_landmark.T).T)).T 

# 	## Normalize using PCA, if requested
# 	if (normalize):
# 		m = Y.shape[0]
# 		Y_hat = Y.T @ (np.eye(m) - (1.0/m)*np.ones(shape=(m,m)))
# 		_, U = np.linalg.eigh(Y_hat @ Y_hat.T) # Note: Y * Y.T == (k x k) matrix
# 		Y = (U.T @ Y_hat).T
# 	return(Y)
# %%
