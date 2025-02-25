# pyiron_lammps

[![Pipeline](https://github.com/pyiron/pyiron_lammps/actions/workflows/pipeline.yml/badge.svg)](https://github.com/pyiron/pyiron_lammps/actions/workflows/pipeline.yml)
[![codecov](https://codecov.io/gh/pyiron/pyiron_lammps/graph/badge.svg?token=OeZVIJ9vyW)](https://codecov.io/gh/pyiron/pyiron_lammps)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/pyiron/pyiron_lammps/main?urlpath=%2Fdoc%2Ftree%2Fnotebooks%2Fexample.ipynb)

The `pyiron_lammps` packages provides primarily two functions. A `write_lammps_structure()` function to write an `ase.atoms.Atoms`
structure to an LAMMPS data file and a `parse_lammps_output_files()` function to parse the `log.lammps`, `dump.out` and
`dump.h5` files from the LAMMPS `thermo` and `dump` commands. 

## Installation 
The `pyiron_lammps` package is distributed via both [pypi](https://pypi.org/project/pyiron-lammps/):
```
pip install pyiron-lammps
```
and [conda-forge](https://anaconda.org/conda-forge/pyiron_lammps):
```
conda install -c conda-forge pyiron_lammps
```

## Write LAMMPS structure 
The `write_lammps_structure()` function is designed to write an `ase.atoms.Atoms` structure to an LAMMPS data file:
```python
from pyiron_lammps import write_lammps_structure

write_lammps_structure(
    structure,
    potential_elements,
    units="metal",
    file_name="lammps.data",
    working_directory=None,
)
```
The `structure` parameter refers to the `ase.atoms.Atoms` structure and the `potential_elements` refers to the list of 
elements implemented in the specific interatomic potential. For example the [NiAlH_jea.eam.alloy](https://github.com/lammps/lammps/blob/develop/potentials/NiAlH_jea.eam.alloy)
potential implements the elements `Ni`, `Al` and `H`, so when writing a structure for a simulation with this potential 
the `potential_elements=["Ni", "Al", "H"]`. It is important to maintain the order of the elements as LAMMPS internally 
references the elements based on their index, starting from one. The `units` parameter refers to the LAMMPS internal 
[units](https://docs.lammps.org/units.html) to convert the `ase.atoms.Atoms` object which is defined in Angstrom to the 
length scale of the LAMMPS simulation. Finally, `file_name` parameter and the current working directory `working_directory` 
parameter are designed to select the location where the LAMMPS structure should be written. With the default parameters 
the LAMMPS structure is written in the `lammps.data` file in the current directory. 

## Parse LAMMPS output
In addition to writing the LAMMPS input structure `pyiron_lammps` also provide the `parse_lammps_output_files()` function
to parse the LAMMPS output files, namely the `log.lammps`, `dump.out` and `dump.h5` files:
```python
from pyiron_lammps import parse_lammps_output_files

parse_lammps_output_files(
    working_directory,
    structure,
    potential_elements,
    units="metal",
    dump_h5_file_name="dump.h5",
    dump_out_file_name="dump.out",
    log_lammps_file_name="log.lammps",
)
```
In analogy to the `write_lammps_structure()` function the `working_directory` parameter refers to the directory which 
contains the output files. The `structure` parameter reefers to the `ase.atoms.Atoms` object which should be used as 
template to parse the structure from the dump files. This structure is again required as LAMMPS internally references 
elements only by an index, so the template structure is required to map the elements from the interatomic potential back
to the elements of the `ase.atoms.Atoms` object. In the same way the `potential_elements` refers to the list of 
elements implemented in the specific interatomic potential. The `units` parameter refers to the LAMMPS internal 
[units](https://docs.lammps.org/units.html) to convert the `ase.atoms.Atoms` object which is defined in Angstrom to the 
length scale of the LAMMPS simulation. Finally, the parameters `dump_h5_file_name`, `dump_out_file_name` and `log_lammps_file_name`
refer to the output file names. 

For the `dump.out` file the following LAMMPS `dump` command should be added to the LAMMPS input file:
```
dump 1 all custom 100 dump.out id type xsu ysu zsu fx fy fz vx vy vz
dump_modify 1 sort id format line "%d %d %20.15g %20.15g %20.15g %20.15g %20.15g %20.15g %20.15g %20.15g %20.15g"
```

For the `log.lammps` file the following LAMMPS `thermo` command should be added to the LAMMPS input file:
```
thermo_style custom step temp pe etotal pxx pxy pxz pyy pyz pzz vol
thermo_modify format float %20.15g
thermo 100
```

## Usage 
Currently, the `pyiron_lammps` parser is primarily used in the [pyiron_atomistics](https://github.com/pyiron/pyiron_atomistics) 
package and its successor the [atomistics](https://github.com/pyiron/atomistics) package to provide a simple LAMMPS 
parser. It only depends on `ase`, `numpy`, `pandas` and `scipy` and has an optional dependency on `h5py` to parse the 
LAMMPS [h5md](https://docs.lammps.org/dump_h5md.html) format. 
