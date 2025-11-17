from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
import os

# Get Eigen include directory from environment or use default
eigen_include = os.environ.get('EIGEN3_INCLUDE_DIR', '../../third_party/eigen-3.4.0')

ext_modules = [
    Pybind11Extension(
        "fasttimes",
        ["fasttimes/fasttimes.cpp"],  # Adjust filename as needed
        include_dirs=[eigen_include],
        extra_compile_args=['/O2'] if os.name == 'nt' else ['-O3', '-march=native'],
        language='c++'
    ),
]

setup(
    name="fasttimes",
    version="0.1.0",
    author="Your Name",
    description="Fast time series operations with C++ and Eigen",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.7",
)
