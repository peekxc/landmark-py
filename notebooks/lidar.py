# %%
import numpy as np
import laspy as lp
import open3d as o3

# %%
## needs lazrs
laz_file = "/Users/mpiekenbrock/Downloads/ma2021_cent_east_Job1006340/ma2021_cent_east_Job1006340.laz"
pc = lp.read(laz_file)
coords = pc.xyz
# laz_file = "/Users/mpiekenbrock/Downloads/ma2021_cent_east_Job1005651/Job1005651_42071_37_07.laz"
# with pylas.open(laz_file) as fh:
#   print('Points from Header:', fh.header.point_count)
#   las = fh.read()
# pc = lp.read("/Users/mpiekenbrock/Downloads/ma2021_cent_east_Job1006340/ma2021_cent_east_Job1006340.laz")
# point_cloud = lp.read("/Users/mpiekenbrock/Downloads/ma2021_cent_east_Job1005651/Job1005651_42071_37_07.laz")


# point_cloud = lp.file.File(las_file, mode="r")
# coords = np.loadtxt("/Users/mpiekenbrock/Downloads/ma2021_cent_east_Job1005415/Job1005415_42071_33_04.txt", skiprows=1, delimiter=",")
normalize = lambda x: (x - np.min(x))/(np.max(x) - np.min(x))
xyz = coords[:,:3].copy()
xyz[:,0] = normalize(xyz[:,0])
xyz[:,1] = normalize(xyz[:,1])
xyz[:,2] = normalize(xyz[:,2])*0.40

# Intensity,Class,Time
pcl = o3.geometry.PointCloud()
pcl.points = o3.utility.Vector3dVector(xyz)
# pcd.colors = o3.utility.Vector3dVector(colors / 65535)
o3.visualization.draw_geometries([pcl])



# points = np.vstack((point_cloud.x, point_cloud.y, point_cloud.z)).transpose()
# colors = np.vstack((point_cloud.red, point_cloud.green, point_cloud.blue)).transpose()
# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(points)
# pcd.colors = o3d.utility.Vector3dVector(colors / 65535)

from landmark import landmarks
x,y,z = xyz.T
xi = np.digitize(x, np.linspace(0,1,10,endpoint=True)) - 1
yi = np.digitize(y, np.linspace(0,1,10,endpoint=True)) - 1

from itertools import product
from array import array
new_coords = array('f')
for i,j in product(range(10), range(10)):
  local_coords = xyz[(xi == i) & (yi == j)]
  if np.size(local_coords) > 0:
    ind = landmarks(local_coords, k=5000)
    new_coords.extend(local_coords[ind].flatten())
  print(f"({i},{j})")

# xyz_reduced = np.array(new_coords).reshape((len(new_coords) // 3, 3))
xyz_reduced = np.load("/Users/mpiekenbrock/landmark-py/notebooks/xyz_reduced.npz")['arr_0']

pcl = o3.geometry.PointCloud()
pcl.points = o3.utility.Vector3dVector(xyz)
pcl.points = o3.utility.Vector3dVector(xyz_reduced)
# o3.visualization.draw_geometries([pcl])


for p in [0.1, 0.5, 0.999, 1.0, 1.5, 2.0, 2.5, 3.0]:
  d1 = np.array([np.sum(np.abs(x1-x2)**p) for x1, x2 in it.combinations(X, 2)])
  d2 = np.array([np.sum(np.abs(x1-x2)**p)**(1/p) for x1, x2 in it.combinations(X, 2)])
  assert np.all(np.argsort(d1) == np.argsort(d2))

import json
viewpoint_file = "viewpoint.json"

def save_viewpoint(vis):
  ctr = vis.get_view_control()
  params = ctr.convert_to_pinhole_camera_parameters()
  o3.io.write_pinhole_camera_parameters(viewpoint_file, params)
  print("Viewpoint saved to", viewpoint_file)
  return False

def load_viewpoint(vis):
  ctr = vis.get_view_control()
  params = o3.io.read_pinhole_camera_parameters(viewpoint_file)
  ctr.convert_from_pinhole_camera_parameters(params, True)
  print("Viewpoint loaded from", viewpoint_file)
  return False

vis = o3.visualization.VisualizerWithKeyCallback()
vis.create_window()
vis.add_geometry(pcl)
vis.register_key_callback(ord("S"), save_viewpoint)
vis.register_key_callback(ord("L"), load_viewpoint)
vis.run()


# np.savez_compressed("/Users/mpiekenbrock/landmark-py/notebooks/xyz.npz", xyz)
# np.savez_compressed("/Users/mpiekenbrosck/landmark-py/notebooks/xyz_reduced.npz", xyz_reduced)

## Unreduced: 3,844,860 points
## Reduced: 405,028 points
## About a 90% reduction!
# 405028/3844860
