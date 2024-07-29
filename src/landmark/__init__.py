"""Landmark package for finding metric-center approximations."""
import importlib.metadata
from .k_center import landmarks  # noqa: F401

__version__ = importlib.metadata.version("landmark")


## Based on Numpy's usage: https://github.com/numpy/numpy/blob/v1.25.0/numpy/lib/utils.py#L75-L101
def get_include():
	"""Return the directory that contains the packages *.h header files.

	Extension modules that need to compile against the packages exported header files should use this
	function to locate the appropriate include directory.

	Notes:
	  When using `distutils`, for example in `setup.py`:
	    ```python
	    ...
	    Extension('extension_name', ..., include_dirs=[landmark.get_include()])
	    ...
	    ```
	  Or with `meson-python`, for example in `meson.build`:
	    ```meson
	    ...
	    run_command(py, ['-c', 'print(landmark.get_include())', check : true).stdout().strip()
	    ...
	    ```
	"""
	import os

	d = os.path.join(os.path.dirname(__file__), "include")
	return d
