from setuptools import setup
from Cython.Build import cythonize
import numpy

"""
Necessary for Cython to build the Cythonized Python code
This file is called when running: 
	python setup.py build_ext --inplace
"""

setup(
    ext_modules = cythonize(("torus_model.pyx", "sheet_model.pyx", "torus_model_random.pyx")),
    include_dirs=[numpy.get_include()],
    zip_safe=False
)