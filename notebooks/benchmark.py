import numpy as np
from greedypermutation import clarksongreedy
from landmark import landmarks
from scipy.spatial.distance import cdist

def greedy(P, distance='sqeuclidean'):
  S = cdist(P, P, distance)
  D = S[0]
  j = 0
  for _ in P:
    yield P[j]
    j = D.argmax()
    D = np.minimum(S[j], D)

X = np.random.uniform(size=(1500,2))

import timeit
timeit.timeit(lambda: list(greedy(X)), number=150)

timeit.timeit(lambda: landmarks(X, k=1500), number=150)
