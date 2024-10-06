project(
  'landmark', 'cpp', 
  version: run_command(['src/landmark/__version__.py'], check: true).stdout().strip(),
  license: 'Apache-2.0',
  meson_version: '>= 1.2.0',
  default_options : [
    'buildtype=debugoptimized', 
    'warning_level=2',
    'pkgconfig.relocatable=true', 
    'cpp_std=c++17', 
    'python.bytecompile=2'
  ]
)
pkg_name = meson.project_name()
includes = []

## Import python kernel 
py_mod = import('python')
py = py_mod.find_installation(pure: true, required: true, modules: ['numpy']) # python_installation object
message('Python path =' + py.full_path() + ', version: ' + py.version())
message('Numpy version =' + run_command(py, ['-c', 'import numpy; print(numpy.__version__)'], check: true).stdout().strip())

## Check the python version
if py.language_version().version_compare('< 3.8')
  error('Invalid Python version, only >= 3.8 is supported.')
endif

## Print the C++ compiler + tack on default arguments
cpp = meson.get_compiler('cpp')
message('Compiler = '+cpp.get_id()+', version: '+cpp.version())
_cpp_args = cpp.get_supported_arguments(
  '-Wno-unused-but-set-variable',
  '-Wno-unused-function',
  '-Wno-conversion',
  '-Wno-misleading-indentation',
  '-fPIC', 
  '-fvisibility=hidden',
  '-Wno-deprecated-anon-enum-enum-conversion'
)

## Include directories
incdir_numpy = run_command(py, ['-c', 'import os; os.chdir(".."); import numpy; print(numpy.get_include())'], check : true).stdout().strip()
# incdir_pybind11 = run_command(py, ['-c', 'import os; os.chdir(".."); import pybind11; print(pybind11.get_include())'], check : true).stdout().strip()

## Header includes
includes += include_directories(incdir_numpy)
# includes += include_directories(incdir_pybind11)

fs = import('fs')
if fs.is_dir('extern')
  includes += include_directories('extern')
endif


pythran = find_program('pythran', native: true, version: '>=0.14.0')
_cpp_args += [
	'-DENABLE_PYTHON_MODULE',
	'-D__PYTHRAN__=3',
	# '-DPYTHRAN_BLAS_NONE'
]
_cpp_args += cpp.get_supported_arguments(
  '-Wno-unused-but-set-variable',
  '-Wno-unused-function',
  '-Wno-conversion',
  '-Wno-misleading-indentation',
  '-Wno-unused-lambda-capture',
  '-Wno-unused-parameter', 
	'-Wno-deprecated-declarations'
)

## Compile the package directory
subdir('src' / pkg_name)