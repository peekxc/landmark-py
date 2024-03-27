import numpy as np 

def test_basic():
  from landmark import landmarks
  X = np.random.uniform(size=(10,2))
  ind = landmarks(X, k=5)
  assert isinstance(ind, np.ndarray) and ind.dtype == np.uint64
  assert len(ind) == 5
  assert len(landmarks(X, k = 2*len(X))) == len(X)
  assert all(np.sort(landmarks(X, k = len(X))) == np.arange(len(X)))

def test_equiv():
  from landmark import landmarks
  from scipy.spatial.distance import pdist, squareform
  X = np.random.uniform(size=(50,2))
  assert all(landmarks(pdist(X)) == landmarks(X)), "Point cloud input equivalent on pairwise distance"
  assert all(landmarks(squareform(pdist(X))) == landmarks(X)), "Point cloud input equivalent on distance matrices"
