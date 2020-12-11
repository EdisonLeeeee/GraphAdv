#!/usr/bin/env python
# coding: utf-8
try:
    import graphgallery
except ImportError:
    raise ImportError("No module named 'graphgallery'. Please install it first! Using 'pip install graphgallery -U'.")
    
from setuptools import setup, find_packages
from graphadv import __version__

VERSION = __version__
url = 'https://github.com/EdisonLeeeee/GraphgAdv'

install_requires = [
    'networkx>=2.3',
    'scipy',
    'tensorflow>=2.1.0',
    'numpy',
    'texttable',
    'numba',
    'tqdm',
    'scikit_learn',
    'graphgallery',
]

setup(
    name='graphadv',
    version=VERSION,
    description='Geometric Adversarial Learning Library for TensorFlow and PyTorch',
    author='Jintang Li',
    author_email='cnljt@outlook.com',
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url=url,
    download_url='{}/archive/{}.tar.gz'.format(url, VERSION),
    keywords=[
        'tensorflow',
        'geometric-deep-learning',
        'graph-adversarial-learning',
        'graph-adversarial-attack',
        'graph-robustness'
    ],
    python_requires='>=3.6',
    license="MIT LICENSE",    
    install_requires=install_requires,
    packages=find_packages(exclude=("examples", "imgs")),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)

