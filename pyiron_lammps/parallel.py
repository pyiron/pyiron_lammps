import numpy as np
from ase.atoms import Atoms
from pandas import DataFrame, Series
from pylammpsmpi import LammpsASELibrary

from pyiron_lammps.calculation import (
    optimize_structure,
    calculate_elastic_constants,
    calculate_energy_volume_curve,
)


def _get_lammps_mpi(enable_mpi=True):
    if enable_mpi:
        # To get the right instance of MPI.COMM_SELF it is necessary to import it inside the function.
        from mpi4py import MPI

        return LammpsASELibrary(
            working_directory=None,
            cores=1,
            comm=MPI.COMM_SELF,
            logger=None,
            log_file=None,
            library=None,
            diable_log_file=True,
        )
    else:
        return LammpsASELibrary(
            working_directory=None,
            cores=1,
            comm=None,
            logger=None,
            log_file=None,
            library=None,
            diable_log_file=True,
        )


def _parallel_execution(function, input_parameter_lst, lmp=None, executor=None):
    if executor is None:
        return [
            function(input_parameter=input_parameter + [False, lmp])
            for input_parameter in input_parameter_lst
        ]
    elif executor is not None and lmp is None:
        return list(
            executor.map(
                function,
                [
                    input_parameter + [True, None]
                    for input_parameter in input_parameter_lst
                ],
            )
        )
    elif executor is not None and lmp is not None:
        raise ValueError(
            "The external LAMMPS instance can only be used for serial execution."
        )
    else:
        raise ValueError("The number of cores has to be a positive integer.")


def _optimize_structure_serial(input_parameter):
    structure, potential_dataframe, enable_mpi, lmp = input_parameter
    if lmp is None:
        return optimize_structure(
            lmp=_get_lammps_mpi(enable_mpi=enable_mpi),
            structure=structure,
            potential_dataframe=potential_dataframe,
        )
    else:
        return optimize_structure(
            lmp=lmp,
            structure=structure,
            potential_dataframe=potential_dataframe,
        )


def _calculate_elastic_constants_serial(input_parameter):
    (
        structure,
        potential_dataframe,
        num_of_point,
        eps_range,
        sqrt_eta,
        fit_order,
        minimization_activated,
        enable_mpi,
        lmp,
    ) = input_parameter
    if lmp is None:
        return calculate_elastic_constants(
            lmp=_get_lammps_mpi(enable_mpi=enable_mpi),
            structure=structure,
            potential_dataframe=potential_dataframe,
            num_of_point=num_of_point,
            eps_range=eps_range,
            sqrt_eta=sqrt_eta,
            fit_order=fit_order,
            minimization_activated=minimization_activated,
        )
    else:
        return calculate_elastic_constants(
            lmp=lmp,
            structure=structure,
            potential_dataframe=potential_dataframe,
            num_of_point=num_of_point,
            eps_range=eps_range,
            sqrt_eta=sqrt_eta,
            fit_order=fit_order,
            minimization_activated=minimization_activated,
        )


def _calculate_energy_volume_curve_serial(input_parameter):
    (
        structure,
        potential_dataframe,
        num_points,
        fit_type,
        fit_order,
        vol_range,
        axes,
        strains,
        minimization_activated,
        enable_mpi,
        lmp,
    ) = input_parameter
    if lmp is None:
        return calculate_energy_volume_curve(
            lmp=_get_lammps_mpi(enable_mpi=enable_mpi),
            structure=structure,
            potential_dataframe=potential_dataframe,
            num_points=num_points,
            fit_type=fit_type,
            fit_order=fit_order,
            vol_range=vol_range,
            axes=axes,
            strains=strains,
            minimization_activated=minimization_activated,
        )
    else:
        return calculate_energy_volume_curve(
            lmp=_get_lammps_mpi(enable_mpi=enable_mpi),
            structure=structure,
            potential_dataframe=potential_dataframe,
            num_points=num_points,
            fit_type=fit_type,
            fit_order=fit_order,
            vol_range=vol_range,
            axes=axes,
            strains=strains,
            minimization_activated=minimization_activated,
        )


def optimize_structure_parallel(
    structure, potential_dataframe, lmp=None, executor=None
):
    input_parameter_lst, output_as_lst = combine_structure_and_potential(
        structure=structure, potential_dataframe=potential_dataframe
    )
    if output_as_lst:
        return _parallel_execution(
            function=_optimize_structure_serial,
            input_parameter_lst=input_parameter_lst,
            lmp=lmp,
            executor=executor,
        )
    else:
        return _parallel_execution(
            function=_optimize_structure_serial,
            input_parameter_lst=input_parameter_lst,
            lmp=lmp,
            executor=executor,
        )[0]


def calculate_elastic_constants_parallel(
    structure,
    potential_dataframe,
    num_of_point=5,
    eps_range=0.005,
    sqrt_eta=True,
    fit_order=2,
    minimization_activated=False,
    executor=None,
    lmp=None,
):
    combo_lst, output_as_lst = combine_structure_and_potential(
        structure=structure, potential_dataframe=potential_dataframe
    )
    input_parameter_lst = [
        [
            s,
            p,
            num_of_point,
            eps_range,
            sqrt_eta,
            fit_order,
            minimization_activated,
        ]
        for s, p in combo_lst
    ]
    if output_as_lst:
        return _parallel_execution(
            function=_calculate_elastic_constants_serial,
            input_parameter_lst=input_parameter_lst,
            executor=executor,
            lmp=lmp,
        )
    else:
        return _parallel_execution(
            function=_calculate_elastic_constants_serial,
            input_parameter_lst=input_parameter_lst,
            executor=executor,
            lmp=lmp,
        )[0]


def calculate_energy_volume_curve_parallel(
    structure,
    potential_dataframe,
    num_points=11,
    fit_type="polynomial",
    fit_order=3,
    vol_range=0.05,
    axes=("x", "y", "z"),
    strains=None,
    minimization_activated=False,
    executor=None,
    lmp=None,
):
    combo_lst, output_as_lst = combine_structure_and_potential(
        structure=structure, potential_dataframe=potential_dataframe
    )
    input_parameter_lst = [
        [
            s,
            p,
            num_points,
            fit_type,
            fit_order,
            vol_range,
            axes,
            strains,
            minimization_activated,
        ]
        for s, p in combo_lst
    ]
    if output_as_lst:
        return _parallel_execution(
            function=_calculate_energy_volume_curve_serial,
            input_parameter_lst=input_parameter_lst,
            executor=executor,
            lmp=lmp,
        )
    else:
        return _parallel_execution(
            function=_calculate_energy_volume_curve_serial,
            input_parameter_lst=input_parameter_lst,
            executor=executor,
            lmp=lmp,
        )[0]


def combine_structure_and_potential(structure, potential_dataframe):
    if isinstance(structure, (list, np.ndarray)):
        if isinstance(potential_dataframe, (list, np.ndarray)):
            if len(structure) == len(potential_dataframe):
                return [[s, p] for s, p in zip(structure, potential_dataframe)], True
            else:
                raise ValueError(
                    "Input lists have len(structure) != len(potential_dataframe) ."
                )
        elif isinstance(potential_dataframe, (DataFrame, Series)):
            return [[s, potential_dataframe] for s in structure], True
        else:
            raise TypeError(
                "potential_dataframe should either be an pandas.DataFrame object or a list of those. "
            )
    elif isinstance(structure, Atoms):
        if isinstance(potential_dataframe, (list, np.ndarray)):
            return [[structure, p] for p in potential_dataframe], True
        elif isinstance(potential_dataframe, (DataFrame, Series)):
            return [[structure, potential_dataframe]], False
        else:
            raise TypeError(
                "potential_dataframe should either be an pandas.DataFrame object or a list of those. "
            )
    else:
        raise TypeError(
            "structure should either be an ase.atoms.Atoms object or a list of those."
        )
