## Extension modules
# extensions = [ '_maxmin']
# foreach ext: extensions
#   py.extension_module(
#     ext,
#     ext + '.cpp',
#     include_directories: includes,
#     install: true, 
#     cpp_args: _cpp_args
#   )
# endforeach

python_sources = [
  '__init__.py', 
  '__version__.py', 
  'datasets.py',
  'k_center.py', 
	'maxmin.py'
]

pythran_gen = generator(
	pythran,
	output : '@BASENAME@.cpp',
	arguments : ['-E', '@INPUT@', '-o', '@OUTPUT@'],
)
# pythran_cmd = [py.path(), '-m', 'pythran.run', '-E', '@INPUT@', '-o', '@OUTPUT@']
pythranrc = join_paths(meson.project_source_root(), '.pythranrc')
pythran_env = environment()
pythran_env.set('PYTHRANRC', pythranrc)
pythran_modules = ['maxmin']
# maxmin_cpp = custom_target(
#   'maxmin_cpp',
#   output : 'maxmin.cpp',
#   input: 'maxmin.py',
#   command: pythran_cmd,
#   env: pythran_env,
#   depend_files: [pythranrc],
# )
pythran_cppflags = run_command(py.path(), '-m', 'pythran.config', '--cflags', check:true, env:pythran_env).stdout().strip().split()
pythran_ldflags = run_command(py.path(), '-m', 'pythran.config', '--libs', check:true, env:pythran_env).stdout().strip().split()

foreach pythran_ext : pythran_modules
	message('Processing pythran module: ' + pythran_ext)
	py.extension_module(
		pythran_ext,
		pythran_gen.process(pythran_ext + '.py'),
		cpp_args: pythran_cppflags + _cpp_args,
		# link_args: pythran_ldflags,
		install: true,
		subdir: 'landmark'
	)
endforeach

## Print install directory
message('Source install dir = '+py.get_install_dir())
py.install_sources(
  python_sources,
  subdir: pkg_name, 
  pure: false       # Will be installed next to binaries
)