# %% Imports
import numpy as np
from bokeh.plotting import figure, show 
from bokeh.io import output_notebook
from bokeh.layouts import row, column
output_notebook(verbose=False, hide_banner=True)

# %% Show landmarks on shape data set(s)
from landmark import landmarks
from landmark.datasets import load_shape
X = load_shape("aggregation")[:,:2]
K = 40
ind = landmarks(X, k = K)

ps = figure(width=350, height=350, title="Original data + Landmarks")
ps.scatter(*X.T, color='lightblue', size=3, line_color='black', line_width=0.5)
ps.scatter(*X[ind].T, color='red', size=6, line_color='white')
show(ps)

# %% Show packing / coverage guarentees 
ind, radii = landmarks(X, k = K, radii = True)

## Show coverage of the union 
p = figure(width=350, height=350, title="Coverage guarantee")
p.circle(*X[ind].T, radius=radii[-1], fill_color='yellow', fill_alpha=0.15)
p.scatter(*X.T, color='lightblue', size=3, line_color='black', line_width=0.5)
p.scatter(*X[ind].T, color='red', size=6, line_color='white')
show(p)

## Show packing of the union 
q = figure(width=350, height=350, title="Packing guarantee")
q.circle(*X[ind].T, radius=radii[-1] / 2.0, fill_color='orange', fill_alpha=0.15)
q.scatter(*X.T, color='lightblue', size=3, line_color='black', line_width=0.5)
q.scatter(*X[ind].T, color='red', size=6, line_color='white')
show(row(p, q))

# %% Can use essentially any metric, though coverage/packing properties may not hold
from scipy.spatial.distance import pdist, squareform
X_dist = pdist(X - X.mean(axis=0), metric="cityblock")
ind = landmarks(X_dist, k = K)

## Show different metrics
pc = figure(width=350, height=350, title="Landmarks with L1 metric")
pc.scatter(*X.T, color='lightblue', size=3, line_color='black', line_width=0.5)
pc.scatter(*X[ind].T, color='red', size=6, line_color='white')
show(pc)

# %% Figure from readme 
for f in [ps, p, q]:
  f.output_backend = 'svg'
  f.toolbar_location = None
  f.xaxis.visible = False
  f.yaxis.visible = False
  f.title.align = 'center'
show(row(ps, p, q))

from bokeh.io import export_svg
export_svg(row(ps, p, q), filename="images/k_center.svg")
