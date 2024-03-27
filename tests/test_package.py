import numpy as np 
import skpackage

def test_package():
  from skpackage import _prefixsum # native extension module
  true_sum = np.cumsum(range(10))
  test_sum = _prefixsum.prefixsum(np.arange(10))
  assert np.all(test_sum == true_sum)
