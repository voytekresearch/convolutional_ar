"""Setup script."""

import os
from setuptools import setup, find_packages

# Get the current version number from inside the module
with open(os.path.join('convolutional_ar', 'version.py')) as version_file:
    exec(version_file.read())

# Load the long description from the README
with open('README.md') as readme_file:
    long_description = readme_file.read()

# Load the required dependencies from the requirements file
with open("requirements.txt") as requirements_file:
    install_requires = requirements_file.read().splitlines()

setup(
    name = 'convolutional_ar',
    version = __version__,
    description = 'Convolutional autoregressive models.',
    long_description = long_description,
    python_requires = '>=3.6',
    author = 'The Voytek Lab',
    author_email = 'voyteklab@gmail.com',
    maintainer = 'Ryan Hammonds',
    maintainer_email = 'rphammonds@ucsd.edu',
    url = 'https://github.com/voytekresearch/convolutional_ar',
    packages = find_packages(),
    license = 'Apache License, 2.0',
    classifiers = [
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: MacOS',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11'
    ],
    platforms = 'any',
    project_urls = {},
    download_url = 'https://github.com/voytekresearch/convolutional_ar/releases',
    keywords = ['convolutional', 'autoregressive', 'image', 'texture'],
    install_requires = install_requires,
    tests_require = ['pytest', 'pytest-cov'],
    extras_require = {
        'tests'   : ['pytest', 'pytest-cov'],
        'all'     : ['pytest', 'pytest-cov']
    }
)
