import os
import sys
import numpy
from setuptools import setup, Extension
from numpy.distutils.misc_util import get_info

# Get LAPACK info
try:
    lapack_info = get_info('lapack_opt')
    if not lapack_info:
        lapack_info = get_info('lapack')
except:
    # Fallback to default configuration
    lapack_info = {
        'libraries': ['lapack', 'blas'] if sys.platform != 'win32' else ['mkl_rt'],
        'library_dirs': []
    }

# Define the extension module
include_dirs = [numpy.get_include()]
extra_compile_args = []
extra_link_args = []

# Add platform-specific configurations
if sys.platform == 'win32':
    extra_compile_args.extend(['/O2'])
    if 'libraries' not in lapack_info:
        lapack_info['libraries'] = ['mkl_rt']
    if 'library_dirs' not in lapack_info:
        lapack_info['library_dirs'] = []
else:  # Unix-like
    extra_compile_args.extend(['-O3'])
    if 'libraries' not in lapack_info:
        lapack_info['libraries'] = ['lapack', 'blas']
    if 'library_dirs' not in lapack_info:
        lapack_info['library_dirs'] = []

ext_modules = [Extension(
    name='lapack_dgeqp3rk',
    sources=sources,
    include_dirs=include_dirs,
    libraries=lapack_info.get('libraries', []),
    library_dirs=lapack_info.get('library_dirs', []),
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
)]

# Setup configuration
setup(
    name='lapack_dgeqp3rk',
    ext_modules=ext_modules,
    install_requires=['numpy>=1.16.0'],
)