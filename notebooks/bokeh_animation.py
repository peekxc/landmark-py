# %% Imports 
import numpy as np
from scipy.spatial.distance import pdist, cdist, squareform
from bokeh.plotting import figure, show
from bokeh.models import Button, CustomJS, Slider, ColumnDataSource, RangeSlider
from bokeh.io import output_notebook
from bokeh.layouts import row, column
from landmark import landmarks
from landmark.datasets import load_shape
# output_notebook()

# %% Get data 
X = load_shape("aggregation")[:,:2]
# X = np.random.uniform(size=(50, 2))
K = 15
ind, radii = landmarks(X, k = len(X), radii=True)
X = X[ind]

# %% Make app with interactive K + covering 
## Setup resources necessary for slider
D = ColumnDataSource(dict(x=X[:,0], y=X[:,1], point_color=np.repeat('red', len(X)), radius=np.repeat(0.05, len(X)), ir=radii))
ps = figure(width=350, height=350, title="Original data + Landmarks")
cg = ps.circle(x='x', y='y', radius='radius', fill_color='yellow', fill_alpha=0.15, line_color='black', source=D)
sg = ps.scatter(x='x', y='y', color='point_color', source=D)
# show(ps)

## K -slider
k_slider = Slider(start=2, end=len(X), value=15, step=1, title="Number of Landmarks")
cg_callback = CustomJS(args=dict(source=D), code="""
  const k = cb_obj.value
  console.log(source.data)
  const R = source.data.radius
  source.data = {
    x: source.data.x, 
    y: source.data.y, 
    point_color: Array.from(R, (r, i) => (i < k ? 'red' : 'gray')), 
    radius: Array.from(R, (r, i) => (i < k ? source.data.ir[k-1] : 0.0)), 
    ir: source.data.ir
  }
""")
k_slider.js_on_change('value', cg_callback)
# k_slider.js_on_change('value', sg_callback)


# source.data = { x, y, color, radius, ir }
show(column(k_slider, ps))

## Epsilon-ball slider
# eps_slider = Slider(start=1e-8, end=np.max(pdist(X))/2, value=1e-8, step=0.0001, title="Radius")
# eps_slider.js_link('value', cg.glyph, 'radius')
# show(column(eps_slider, ps))



# source.data = { x, y }
# const x = source.data.x
# const L = source.data.landmark
# const y = Array.from(L, (l) => Math.pow(x, f))




# callback = CustomJS(args=dict(xr=plot.x_range, yr=plot.y_range, slider=slider), code="""
# export default (args, obj, data, context) => {
#     count += 1
#     console.log(`CustomJS was called ${count} times`)

#     const a = args.slider.value
#     const b = obj.value

#     const {xr, yr} = args
#     xr.start = my_function(a)
#     xr.end = b
# }""")



# %%
