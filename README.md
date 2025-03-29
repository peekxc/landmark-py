# Landmark 

`landmark` is a Python package that constructs a sequence of _landmarks_ $L_k = (\{x_1, x_2, \dots, x_k \})$ from a point set $X \subset \mathbb{R}^d$ or metric space $(X, d_X)$ via [furthest-first traversal](https://en.wikipedia.org/wiki/Farthest-first_traversal):

$$ x_i = \mathop{\mathrm{arg max}}\limits_{x \in X} \mathop{} d_X(x, L_{i − 1}) $$

The resulting sequence, also called the _greedy permutation_, satisfy a number of coverage and packing properties. Below is an example a data set $X$ (blue points), some sample landmarks $L$ (red), along with the coverage (yellow) and packing (orange) properties they obey. 

![Landmarks example](docs/images/k_center.svg)

The landmarks themselves $L_k$ can be used to approximate solutions to hard problems, including the [traveling saleman problem](https://ieeexplore.ieee.org/document/9001738), the [metric k-center clustering problem](https://en.wikipedia.org/wiki/Metric_k-center), [geometric filtration sparsification problem](https://donsheehy.net/research/cavanna15geometric.pdf), and various [proximity searching problems](https://donsheehy.net/research/chubet23proximity.pdf).

## Installation 

The package can be installed with [pip](https://packaging.python.org/en/latest/guides/tool-recommendations/#installing-packages): 

```{bash}
python -m pip install scikit-landmark
```

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

For other usages, see the [documentation](https://peekxc.github.io/landmark-py/greedy_perm.html). 
