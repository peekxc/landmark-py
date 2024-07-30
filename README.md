# Landmark 

`landmark` is a Python package that constructs _landmarks_ $L_k = \{x_1, x_2, \dots, x_k \}$ from a point set $X \subset \mathbb{R}^d$ or metric space $(X, d_X)$.
<!-- `landmark` is a Python package that constructs _landmarks_ $L^\ast \subset X$ from a point set $X \subset \mathbb{R}^d$ or a metric space $(X, d_X)$ that approximate the [metric k-center problem](https://en.wikipedia.org/wiki/Metric_k-center): 

$$ L^\ast \triangleq \mathop{\mathrm{argmin}}\limits_{\substack{L \subseteq X : \lvert L \rvert = k}} \ \max_{x \in X} d_X(x, L)$$ -->

Below is an example a data set $X$ (blue points), some sample landmarks $L$ (red), along with the coverage (yellow) and packing (orange) properties they obey. 

![Landmarks example](docs/images/k_center.svg)

## Installation 

The package can be installed with [pip](https://packaging.python.org/en/latest/guides/tool-recommendations/#installing-packages): 

```{bash}
python -m pip install scikit-landmark
```

Alternatively, both the source distribution and wheels are available on [PyPI](https://pypi.org/project/scikit-landmark/)  for distributing and offline use.

## Usage 

Given a point cloud $X \in \mathbb{R}^{n \times d}$ represented as a numpy matrix with $n$ points in $d$ dimensions, the indices of the landmarks can be found with the `landmarks` function:

```python
from landmark import landmarks
X = np.random.uniform(size=(50,2))
ind = landmarks(X, k = 10) ## Finds the indices of 25 landmarks
```

The first $k$-indices of `ind` are equivalent to the $k$-th prefix of the [greedy permutation](https://www.youtube.com/watch?v=xWuq1aXHLdU). You can get their covering radii and their predecessors by specifying `full_output=True`:

```python
ind, info = landmarks(X, k = 10, full_output = True)
print(ind)                  ## prefix indices
print(info['radii'])        ## insertion radii 
print(info['predecessors']) ## predecessor map 
```