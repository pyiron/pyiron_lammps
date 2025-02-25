from pyiron_lammps.output import parse_lammps_output as parse_lammps_output_files
from pyiron_lammps.structure import write_lammps_datafile as write_lammps_structure

from . import _version

DUMP_COMMANDS = [
    "dump 1 all custom 100 dump.out id type xsu ysu zsu fx fy fz vx vy vz\n",
    'dump_modify 1 sort id format line "%d %d %20.15g %20.15g %20.15g %20.15g %20.15g %20.15g %20.15g %20.15g %20.15g"\n',
]

THERMO_COMMANDS = [
    "thermo_style custom step temp pe etotal pxx pxy pxz pyy pyz pzz vol\n",
    "thermo_modify format float %20.15g\n",
    "thermo 100\n",
]

__version__ = _version.get_versions()["version"]
__all__ = [
    "parse_lammps_output_files",
    "write_lammps_structure",
]
