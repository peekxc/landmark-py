import numpy as np 

def test_basic():
  from landmark import landmarks
  np.random.seed(1234)
  X = np.random.uniform(size=(10,2))
  ind = landmarks(X, k=5)
  assert isinstance(ind, np.ndarray) and ind.dtype == np.uint64
  assert len(ind) == 5
  assert len(landmarks(X, k = 2*len(X))) == len(X)
  assert all(np.sort(landmarks(X, k = len(X))) == np.arange(len(X)))

def test_equiv():
  from landmark import landmarks
  from scipy.spatial import distance
  from scipy.spatial.distance import pdist, squareform
  np.random.seed(1234)
  X = np.random.uniform(size=(50,2))
  for metric in ['euclidean', 'cityblock', 'chebychev']:
    lm = landmarks(X, metric=metric)
    lm_pdist = landmarks(pdist(X, metric=metric))
    lm_cdist = landmarks(squareform(pdist(X, metric=metric)))
    assert all(lm == lm_pdist), f"Point cloud input not equivalent on pairwise distance ({metric})"
    assert all(lm_pdist == lm_cdist), f"Point cloud input not equivalent on distance matrices ({metric})"
  lm = landmarks(X, metric="minkowski", p=3.5)
  lm_pdist = landmarks(pdist(X, metric="minkowski", p=3.5))
  assert all(lm == lm_pdist), f"Point cloud input not equivalent on pairwise distance ({metric})"

def test_predecessor():
  from landmark import landmarks
  from scipy.spatial.distance import cdist
  np.random.seed(1234)
  X = np.random.uniform(size=(150,2))
  ind, info = landmarks(X, k=150, full_output=True)
  
  ## The predecessor mapping T : P \ {p0} → P maps each point p in P to the closest point in the prefix Pi
  pred_map = [0]
  for i, lm in enumerate(ind):
    if i > 0:
      pred_map.append(ind[np.argmin(cdist(X[lm][np.newaxis,:], X[ind[:i]]))])
  assert np.all(info['predecessors'] == np.array(pred_map)), "Predecessor map wrong"

  ## Check the insertion radii 
  insertion_radii = np.array([np.linalg.norm(X[lm] - X[p]) for lm, p in zip(ind, info['predecessors'])])
  insertion_radii[0] = np.inf
  assert np.allclose(insertion_radii, info['radii']), "Insertion radii wrong"