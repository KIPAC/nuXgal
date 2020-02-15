#!/usr/bin/env python
from setuptools import setup, find_packages
from version import get_git_version

setup(
    name='KIPAC_nuXgal',
    version=get_git_version(),
    author='Arka Banarjee, Eric Charles, Ke Fang, Yuuki Omori',
    author_email='',
    description='A Python package for analysis of neutrino galaxy cross correlations',
    license='gpl2',
    packages=find_packages(),
    include_package_data=True,
    url="https://github.com/KIPAC/nuXgal",
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GPL2 License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering :: Astronomy',
    ],
    scripts=[],
    entry_points={'console_scripts': [
            'nuXgal_generateICIRFS = scripts.generateICIRFS:main',
            ]},
    install_requires=[
        'numpy >= 1.6.1',
        'astropy >= 3.2.2',
        'matplotlib >= 3.1.1',
        'scipy >= 1.3.1',
        'numba >= 0.45.1',
        'pytest >= 5.2.1',
        'healpy >= 1.12.0',
        'emcee',
        'corner',
    ],
    extras_require=dict(
        all=[],
    ),
)
