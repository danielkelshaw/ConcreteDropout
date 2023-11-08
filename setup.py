# Create a setup.py file for this package to allow installation with pip. The version can be found at condrop.__version__.
from os import path
from codecs import open
from condrop import __version__
from setuptools import setup, find_packages

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md')) as f:
    long_description = f.read()

# Get dependencies from requirements.txt
with open(path.join(here, 'requirements.txt')) as f:
    requirements = f.read().splitlines()

setup(
    name='condrop',
    version=__version__,
    packages=find_packages(),
    description='A PyTorch implementation of Concrete Dropout',
    long_description=long_description,
    license='MIT',
)
