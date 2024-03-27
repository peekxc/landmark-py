# landmark  

`landmark` is a Python package that constructs _landmarks_ $L^\ast \subset X$ from a point set $X \subset \mathbb{R}^d$ or a metric space $(X, d_X)$ that approximate the [metric k-center problem](https://en.wikipedia.org/wiki/Metric_k-center) (also known _k-center clustering_ problem): 

$$ L^\ast \triangleq \mathop{\mathrm{argmin}}\limits_{\substack{L \subseteq X : \lvert L \rvert = k}} \ \max_{x \in X} d_X(x, L)$$

Metric $k$-center is a classic NP-hard problem intrinsically related to many other problems, such as [geometric set cover](https://en.wikipedia.org/wiki/Geometric_set_cover_problem) and [facility location](https://en.wikipedia.org/wiki/Optimal_facility_location); its output is also related to other geometric constructions, like $\epsilon$-[nets](https://en.wikipedia.org/wiki/Delone_set).

![Landmarks example](images/k_center.svg)

<!-- $$ \min\limits_{\substack{L \subseteq X \, : \, \lvert L \rvert = k}} \ \max_{x \in X} \, d_X(x, L)$$ -->
<!-- where $d_X(x, L)$ denotes the Hausdorff distance to the set of landmarks $L$.  -->

## Installation 

Clone and use:

> python -m pip install < landmark-py location >

> Warning: this package is very early-stage, e.g. does not offer wheels on PyPI. Use with care.

## Usage 

Given a point cloud $X \in \mathbb{R}^{n \times d}$ represented as a numpy matrix with $n$ points in $d$ dimensions, the indices of the landmarks can be found with `landmarks`:

```python 
from landmark import landmarks
ind = landmarks(X, k = 25) ## Finds the indices of 25 landmarks
print(ind)
# [0, 32, 45, 16, ...]
```

The first $k$-indices of `ind` are equivalent to the $k$-th prefix of the [greedy permutation](https://www.youtube.com/watch?v=xWuq1aXHLdU). You can get their covering radii by specifying `radii = True`

```python 
from landmark import landmarks
ind, radii = landmarks(X, k = 25, radii = True) ## Finds the indices of 25 landmarks
print(ind, radii)
# [0, 32, 45, 16, ...]
# [inf 31.6024 28.4495 20.3045 ...]
```

For more examples, see [notebook example](notebooks/k_center.py).