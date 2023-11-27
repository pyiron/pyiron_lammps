from . import _version
from atomistics.calculators.lammps import (
    get_potential_dataframe,
)
from pylammpsmpi import LammpsASELibrary
from pyiron_lammps.calculation import (
    optimize_structure,
    calculate_elastic_constants,
    calculate_energy_volume_curve,
)
from pyiron_lammps.parallel import (
    optimize_structure_parallel,
    calculate_elastic_constants_parallel,
    calculate_energy_volume_curve_parallel,
)


def get_lammps_engine(
    working_directory=None,
    cores=1,
    comm=None,
    logger=None,
    log_file=None,
    library=None,
    diable_log_file=True,
):
    return LammpsASELibrary(
        working_directory=working_directory,
        cores=cores,
        comm=comm,
        logger=logger,
        log_file=log_file,
        library=library,
        diable_log_file=diable_log_file,
    )


__version__ = _version.get_versions()["version"]
