#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <prefix_sum.h>

namespace py = pybind11;

template< typename F >
using py_array = py::array_t< F, py::array::f_style | py::array::forcecast >;

PYBIND11_MODULE(_prefixsum, m) {
  m.doc() = "prefix sum module"; 
  m.def("prefixsum", [](const py_array< int >& v) -> py_array< int >{
    auto ps = std::vector(v.size(), 0);
    prefix_sum(v.data(), v.data() + v.size(), ps.begin());
    return py::cast(ps);
  });
}