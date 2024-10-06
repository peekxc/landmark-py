import numpy as np
from landmark import landmarks
from landmark.datasets import load_shape
from bokeh.io import output_notebook
from bokeh.plotting import show, figure
output_notebook()

# X = load_shape("aggregation")[:,:2]
X = np.random.uniform(size=(100,2))
ind, info = landmarks(X, k=50, full_output=True)

## Own greedy tree
from collections import namedtuple
from types import SimpleNamespace
class Node(SimpleNamespace):
  def __init__(self, _id):
    self.id = _id
    self.left = None 
    self.right = None 
    self.parent = None
  
  def __repr__(self) -> str:
    if self.left is None and self.right is None:
      return self.id
    msg = f"{self.id}->"
    msg += f"(L:{self.left.id})" if self.left is not None else ""
    msg += f"(R:{self.right.id})" if self.right is not None else ""
    return msg

# N = [0,1,2,3,4,5]
# P = [None,0,1,0,2,1]
N = ['a',  'b','c','d','e','f']
P = [None, 'a','b','a','c','b']

Counts = { n : -1 for n in np.unique(N) }
Nodes = {}
for n, p in zip(N, P):
  # if n == 'd':
  #   break
  Counts[n] += 1
  cn = str(n) + str(Counts[n])
  Nodes[cn] = Node(cn)
  if p is not None:
    Nodes[cn].parent = Nodes[str(p) + str(Counts[p])]
    Counts[p] += 1
    left_id = str(p) + str(Counts[p])
    Nodes[left_id] = Node(left_id)
    Nodes[cn].parent.left = Nodes[left_id]
    Nodes[cn].parent.right = Nodes[cn]
  print(Nodes)

Nodes['a1']


## The number of leaves in the subtree rooted at any node is also stored in the node
def num_leaves(node):
  if node.left is None and node.right is None: 
    return 1
  num_left = num_leaves(node.left)
  num_right = num_leaves(node.right)
  return num_left + num_right

num_leaves(Nodes['a0'])
num_leaves(Nodes['a1'])
num_leaves(Nodes['a2'])
num_leaves(Nodes['b0'])
num_leaves(Nodes['b1'])

from landmark import landmarks
X = np.random.uniform(size=(50,2))
ind, info = landmarks(X, full_output=True)

N = ind
P = info['predecessors']

Counts = { n : -1 for n in np.unique(N) }
Nodes = {}
for n, p in zip(N, P):
  Counts[n] += 1
  cn = str(n) + str(Counts[n])
  Nodes[cn] = Node(cn)
  if p is not None:
    Nodes[cn].parent = Nodes[str(p) + str(Counts[p])]
    Counts[p] += 1
    left_id = str(p) + str(Counts[p])
    Nodes[left_id] = Node(left_id)
    Nodes[cn].parent.left = Nodes[left_id]
    Nodes[cn].parent.right = Nodes[cn]
  print(Nodes)




# Node = namedtuple("Node", ['id', 'left', 'right', 'parent'])

# Node(ind[0], left=)

# nodes = [Node(id=i) for i,n in enumerate(ind)]
# preds = info['predecessors']
# for i, (node, pred) in enumerate(zip(nodes, preds)):
#   node.left = Node(id=node.id, None, None, node.id)
#   node.parent = pred

  

# class Node:
#   def __init__(self, left: Node, right: Node):
#     self.left = 




import greedypermutation
from greedypermutation import Point
from metricspaces import MetricSpace
from greedypermutation.balltree import greedy_tree
M = MetricSpace([Point(x) for x in X])
T = greedy_tree(M)

T.radius
T.center
# %% Draw the tree 
p = figure(width=300, height=300)
p.scatter(*X.T)

LEVEL = 6
def plot_level(node, level: int = 0):
  if level == LEVEL: 
    p.circle(*node.center, radius=node.radius, fill_alpha=0.0, line_color='black')
    print(f"C:{node.center}, R:{node.radius}")
    return
  if node.left is not None:
    plot_level(node.left, level+1)
  if node.right is not None:
    plot_level(node.right, level+1)

plot_level(T)
show(p)

from greedypermutation import balltree, greedytree, GreedyTree
from greedypermutation.balltree import Ball, MetricSpace, greedy, greedy_tree
# root = BallTree(0)
# leaf = {seed: root}
# for p, q in gp:
#     node = leaf[q]
#     leaf[q] = node.left = BallTree(q)
#     leaf[p] = node.right = BallTree(p)
# root.update()
# return root



