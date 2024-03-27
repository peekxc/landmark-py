# scikit-package template 

This is my own opionated package template for creating python packages geared for scientific computing.
This template is especially geared for creating cross-platform packages that use C++ to build native extension modules. 

In particular, this template uses [quarto](https://quarto.org/) for documentation, [pybind11](https://pybind11.readthedocs.io/en/stable/index.html) for bindings, and [meson-python](https://meson-python.readthedocs.io/en/latest/#) for building. The package template also incorporates a matrix build via [github actions](https://github.com/features/actions) that cross-compiles wheels for each major compiler / architecture using [cibuildwheel](https://cibuildwheel.readthedocs.io/en/stable/setup/) for maximum compatibility. 

Notable choices made in the template: 

- [PEP 518](https://peps.python.org/pep-0518/)-compliance
- Uses the [src layout](https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/)
- Unit tests with [pytest](https://docs.pytest.org/en/7.4.x/)
- Optional `include` folder for exporting packages headers for use in external libraries
- Optional `extern` folder for importing external package headers (e.g. via git submodules)
- GH workflow files for building native wheels on Linux, OS X, and Windows
- Package docs w/ [quarto](https://quarto.org/) + [quartodoc](https://machow.github.io/quartodoc/get-started/overview.html) renders into `docs` for simple GH pages docsite
- Monthly [dependabot](https://docs.github.com/en/code-security/dependabot/working-with-dependabot) updates

The package also contains a simple extension module and some basic meson configuration to jump-start projects that build extension modules. 