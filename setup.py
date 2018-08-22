#!/usr/bin/env python3

import numpy as np
from Cython.Build import cythonize
from setuptools import setup
from setuptools.extension import Extension

with open('requirements.txt', 'r') as f:
    install_requires = f.read().splitlines()

extensions = [
    Extension(
        'pypradana._tools',
        ['pypradana/_tools.pyx'],
        include_dirs=[np.get_include()],
    ),
]

metadata = dict(
    name='pypradana',
    packages=['pypradana'],
    package_dir={
        'pypradana': 'pypradana',
    },
    package_data={
        'pypradana': ['database/*'],
    },
    ext_modules=cythonize(extensions),
    author='Chao Gu',
    author_email='guchao.pku@gmail.com',
    maintainer='Chao Gu',
    maintainer_email='guchao.pku@gmail.com',
    description='Analysis Software for Jefferson Lab PRad Experiment.',
    license='GPL-3.0',
    url='https://github.com/asymmetry/pypradana',
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Operating System :: MacOS',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Utilities',
    ],
    platforms='Any',
    python_requires='>=3.4',
    install_requires=install_requires,
)

setup(**metadata)
