"""pytest unit tests for the landmark package.

Test with: coverage run --source=landmark -m pytest tests && coverage report -m
"""
import numpy as np 
import pytest

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

# def test_datasets():
#   from landmark.datasets import load_shape
#   from landmark.k_center import landmarks
#   X = load_shape("Aggregation")
#   ind = landmarks(X, 15, seed=0)
#   assert np.all(ind == np.array([  0, 501, 754, 707, 435, 105, 580, 750, 303, 203,  62, 165, 403, 517, 621]))

def test_include():
  from landmark import get_include
  include_path = get_include()
  assert isinstance(include_path, str)
  assert include_path[-7:] == "include"

# def test_except():
#   from landmark import landmarks
#   from landmark.datasets import load_shape
#   with pytest.raises(AssertionError) as e_info: 
#     load_shape('testing')
#   X = load_shape("Aggregation")
#   with pytest.raises(ValueError) as e_info: 
#     landmarks('testing', k=5)
#   with pytest.raises(AssertionError) as e_info: 
#     landmarks(X, k=5, metric="canberra")





