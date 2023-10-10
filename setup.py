"""
Setuptools based setup module
"""
from setuptools import setup, find_packages
import versioneer

setup(
    name='pyiron_lammps',
    version=versioneer.get_version(),
    description='pyiron_lammps - calculate material properties for interatomic potentials',
    long_description='http://pyiron.org',

    url='https://github.com/pyiron/pyiron_lammps',
    author='Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department',
    author_email='janssen@mpie.de',
    license='BSD',

    classifiers=['Development Status :: 5 - Production/Stable',
                 'Topic :: Scientific/Engineering :: Physics',
                 'License :: OSI Approved :: BSD License',
                 'Intended Audience :: Science/Research',
                 'Operating System :: OS Independent',
                 'Programming Language :: Python :: 3.8',
                 'Programming Language :: Python :: 3.9',
                 'Programming Language :: Python :: 3.10',
                 'Programming Language :: Python :: 3.11'
                ],

    keywords='pyiron',
    packages=find_packages(exclude=["*tests*", "*docs*", "*binder*", "*conda*", "*notebooks*", "*.ci_support*"]),
    install_requires=[
        'ase==3.22.1',
        'atomistics==0.0.6',
        'numpy==1.26.0',
        'pandas==2.1.1',
        'pylammpsmpi==0.2.4',
        'pympipool==0.7.1',
    ],
    cmdclass=versioneer.get_cmdclass(),
)
