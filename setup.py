#!/usr/bin/env python
from setuptools import setup, find_packages
from KIPAC.nuXgal.version import get_git_version

setup(
    name='KIPAC_nuXgal',
    version=get_git_version(),
    author='Arka Banarjee, Eric Charles, Ke Fang',
    author_email='',
    description='A Python package for analysis of neutrino galaxy cross correlations',
    license='gpl2',
    packages=find_packages(),
    include_package_data=True,
    url="https://github.com/KIPAC/analysis_001",
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GPL2 License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering :: Astronomy',
    ],
    scripts=[],
    entry_points={'console_scripts': []},
    install_requires=[
        'numpy >= 1.6.1',
        'astropy >= 2.0.7',
        'matplotlib >= 1.5.0',
        'scipy >= 0.14',
        'pyyaml',
        'healpy',
    ],
    extras_require=dict(
        all=[],
    ),
)
