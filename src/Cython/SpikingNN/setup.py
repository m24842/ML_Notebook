from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules=[
        Extension("SNN", ["SNN.pyx"], include_dirs=[np.get_include()]),
    ],
)