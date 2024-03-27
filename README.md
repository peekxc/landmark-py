# landmark  

`landmark` is a Python package that constructs _landmarks_ from a point set $X \subset \mathbb{R}^d$ or a metric space $(X, d_X)$ that geometrically approximate the set $X$ . 

The algorithms in the package finds $k$ landmarks $L \subset X$ by approximating the [metric $k$-center problem](https://en.wikipedia.org/wiki/Metric_k-center) (also known _$k$-center clustering_ problem): 

$$ L^\ast(X, k) \triangleq \mathop{\mathrm{arg\,min}}\limits_{\substack{L \subseteq X \, : \, \lvert L \rvert = k}} \ \max_{x \in X} \, d_X(x, L)$$

This above is a classic [NP-hard problem](https://en.wikipedia.org/wiki/List_of_NP-complete_problems) heavily related to many other problems, such as  [geometric set cover](https://en.wikipedia.org/wiki/Geometric_set_cover_problem) and [facility location](https://en.wikipedia.org/wiki/Optimal_facility_location).

![Landmarks example](images/k_center.svg)

<!-- $$ \min\limits_{\substack{L \subseteq X \, : \, \lvert L \rvert = k}} \ \max_{x \in X} \, d_X(x, L)$$ -->
<!-- where $d_X(x, L)$ denotes the Hausdorff distance to the set of landmarks $L$.  -->

> Warning: this package is very early-stage, e.g. does not offer wheels on PyPI. Use with care.