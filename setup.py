import os
from setuptools import setup
from setuptools import find_packages

setup(name='ccGPFA',
      author='Yididiya Nadew <yididiya@iastate.edu>',
      version='0.0.1',
      description='Pytorch implementation of ccGPFA',
      license='MIT',
      install_requires=[
          'numpy==1.24.3', 'torch==2.0.1', 'matplotlib==3.7.1'
      ],
      packages=find_packages())
