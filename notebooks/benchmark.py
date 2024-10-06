import numpy as np
from greedypermutation import clarksongreedy
from landmark import landmarks
from scipy.spatial.distance import cdist
import greedy

def greedy_naive(P, distance='sqeuclidean'):
  S = cdist(P, P, distance)
  D = S[0]
  j = 0
  for _ in P:
    yield j
    j = D.argmax()
    D = np.minimum(S[j], D)

np.random.seed(1234)
X = np.random.uniform(size=(5000,12))

import timeit
from itertools import islice
K = 2500
NI = 10
naive = timeit.timeit(lambda: list(islice(greedy_naive(X), K)), number=NI)/NI
land_pkg = timeit.timeit(lambda: landmarks(X, k=K), number=NI)/NI
land_pythran = timeit.timeit(lambda: greedy.fft(X, k=K-1, eps=-1.0, seed=0), number=NI)/NI
land_pythran2 = timeit.timeit(lambda: greedy.fft2(X, k=K-1, eps=-1.0, seed=0), number=NI)/NI

## Current benchmarks, with SIMD vectorization
# Naive       : 0.24935751450248062
# Package C++ : 0.10345068849856034
# Pythran     : 0.04906970969750546
# Pythran 2   : 0.10209581129602156
print(f"Naive: {naive}")
print(f"Package C++: {land_pkg}")
print(f"Pythran 1: {land_pythran}")
print(f"Pythran 2: {land_pythran2}")
