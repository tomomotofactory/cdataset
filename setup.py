#!/usr/bin/env python
from setuptools import setup, find_packages

setup(
    name='cdataset',
    version='0.1.0',
    description='Classification data set for benchmarking of machine learning',
    author='Tomomitsu Motohashi',
    author_email='tomomotmo1983@gmail.com',
    packages=find_packages(exclude=('input', 'output', 'tool')),
    zip_safe=False,
    include_package_data=True,
    package_data={
      'cdataset': ['data/*/df.arff', 'data/*/target_name.txt'],
    },
    install_requires=[
        'scikit-learn>=0.21.2',
        'pandas>=0.24.2',
        'arff2pandas>=1.0.1'
    ]
)