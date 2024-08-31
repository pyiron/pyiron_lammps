from concurrent.futures import Executor, Future
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
from ase.atoms import Atoms
from pandas import DataFrame, Series
from pylammpsmpi import LammpsASELibrary

from pyiron_lammps.calculation import (
    calculate_elastic_constants,
    calculate_energy_volume_curve,
    optimize_structure,
)


class InProcessExecutor:
    @staticmethod
    def submit(funct, *args, **kwargs):
        f = Future()
        f.set_result(funct(*args, **kwargs))
        return f


def _check_mpi(
    executor: Executor, lmp: LammpsASELibrary
) -> Tuple[bool, Optional[LammpsASELibrary]]:
    if executor is None:
        enable_mpi = False
    elif executor is not None and lmp is None:
        enable_mpi = True
    elif executor is not None and lmp is not None:
        raise ValueError(
            "The external LAMMPS instance can only be used for serial execution."
        )
    else:
        raise ValueError("The number of cores has to be a positive integer.")
    return enable_mpi, lmp


def _get_lammps_mpi(
    lmp: Optional[LammpsASELibrary] = None, enable_mpi: bool = True
) -> LammpsASELibrary:
    """
    Get an instance of LammpsASELibrary for parallel execution using MPI.

    Args:
        enable_mpi (bool): Flag to enable MPI. Default is True.

    Returns:
        LammpsASELibrary: An instance of LammpsASELibrary.

    """
    if lmp is not None:
        return lmp
    elif enable_mpi:
        # To get the right instance of MPI.COMM_SELF it is necessary to import it inside the function.
        from mpi4py import MPI

        return LammpsASELibrary(
            working_directory=None,
            cores=1,
            comm=MPI.COMM_SELF,
            logger=None,
            log_file=None,
            library=None,
            disable_log_file=True,
        )
    else:
        return LammpsASELibrary(
            working_directory=None,
            cores=1,
            comm=None,
            logger=None,
            log_file=None,
            library=None,
            disable_log_file=True,
        )


def optimize_structure_parallel(
    structure: Any,
    potential_dataframe: Union[List[DataFrame], DataFrame],
    lmp: Optional[LammpsASELibrary] = None,
    executor: Optional[Executor] = None,
) -> Union[Any, List[Any]]:
    """
    Optimize the structure in parallel using either a provided executor or the default executor.

    Args:
        structure (Any): The structure to be optimized.
        potential_dataframe (Union[List[DataFrame], DataFrame]): The potential dataframe.
        lmp (Optional[LammpsASELibrary]): An optional instance of LammpsASELibrary for serial execution.
        executor (Optional[Executor]): An optional executor for parallel execution.

    Returns:
        Union[Any, List[Any]]: The result of the optimization.

    """
    input_parameter_lst, output_as_lst = combine_structure_and_potential(
        structure=structure, potential_dataframe=potential_dataframe
    )
    enable_mpi, lmp = _check_mpi(executor=executor, lmp=lmp)
    input_parameter_dict = [
        {
            "lmp": lmp,
            "enable_mpi": enable_mpi,
            "structure": struct,
            "potential_dataframe": pot,
        }
        for [struct, pot] in input_parameter_lst
    ]
    if executor is None:
        executor = InProcessExecutor()
    result_lst = [
        fut.result()
        for fut in [
            executor.submit(optimize_structure, **kwargs)
            for kwargs in input_parameter_dict
        ]
    ]
    if output_as_lst:
        return result_lst
    else:
        return result_lst[0]


def calculate_elastic_constants_parallel(
    structure: Any,
    potential_dataframe: Union[List[DataFrame], DataFrame],
    num_of_point: int = 5,
    eps_range: float = 0.005,
    sqrt_eta: bool = True,
    fit_order: int = 2,
    minimization_activated: bool = False,
    executor: Optional[Executor] = None,
    lmp: Optional[LammpsASELibrary] = None,
) -> Union[Any, List[Any]]:
    """
    Calculate the elastic constants in parallel using either a provided executor or the default executor.

    Args:
        structure (Any): The structure for calculating elastic constants.
        potential_dataframe (Union[List[DataFrame], DataFrame]): The potential dataframe.
        num_of_point (int): The number of points for calculating elastic constants.
        eps_range (float): The range of strain for calculating elastic constants.
        sqrt_eta (bool): Whether to take the square root of eta for calculating elastic constants.
        fit_order (int): The order of polynomial fit for calculating elastic constants.
        minimization_activated (bool): Whether to activate minimization for calculating elastic constants.
        executor (Optional[Executor]): An optional executor for parallel execution.
        lmp (Optional[LammpsASELibrary]): An optional instance of LammpsASELibrary for serial execution.

    Returns:
        Union[Any, List[Any]]: The result of calculating the elastic constants.

    """
    input_parameter_lst, output_as_lst = combine_structure_and_potential(
        structure=structure, potential_dataframe=potential_dataframe
    )
    enable_mpi, lmp = _check_mpi(executor=executor, lmp=lmp)
    input_parameter_dict = [
        {
            "lmp": lmp,
            "enable_mpi": enable_mpi,
            "structure": struct,
            "potential_dataframe": pot,
            "num_of_point": num_of_point,
            "eps_range": eps_range,
            "sqrt_eta": sqrt_eta,
            "fit_order": fit_order,
            "minimization_activated": minimization_activated,
        }
        for [struct, pot] in input_parameter_lst
    ]
    if executor is None:
        executor = InProcessExecutor()
    result_lst = [
        fut.result()
        for fut in [
            executor.submit(calculate_elastic_constants, **kwargs)
            for kwargs in input_parameter_dict
        ]
    ]
    if output_as_lst:
        return result_lst
    else:
        return result_lst[0]


def calculate_energy_volume_curve_parallel(
    structure: Any,
    potential_dataframe: Union[List[DataFrame], DataFrame],
    num_points: int = 11,
    fit_type: str = "polynomial",
    fit_order: int = 3,
    vol_range: float = 0.05,
    axes: Tuple[str, str, str] = ("x", "y", "z"),
    strains: Optional[List[float]] = None,
    minimization_activated: bool = False,
    executor: Optional[Executor] = None,
    lmp: Optional[LammpsASELibrary] = None,
) -> Union[Any, List[Any]]:
    """
    Calculate the energy volume curve in parallel using either a provided executor or the default executor.

    Args:
        structure (Any): The structure for calculating the energy volume curve.
        potential_dataframe (Union[List[DataFrame], DataFrame]): The potential dataframe.
        num_points (int): The number of points for calculating the energy volume curve.
        fit_type (str): The type of fit for calculating the energy volume curve.
        fit_order (int): The order of polynomial fit for calculating the energy volume curve.
        vol_range (float): The range of volume for calculating the energy volume curve.
        axes (Tuple[str, str, str]): The axes for calculating the energy volume curve.
        strains (Optional[List[float]]): The strains for calculating the energy volume curve.
        minimization_activated (bool): Whether to activate minimization for calculating the energy volume curve.
        executor (Optional[Executor]): An optional executor for parallel execution.
        lmp (Optional[LammpsASELibrary]): An optional instance of LammpsASELibrary for serial execution.

    Returns:
        Union[Any, List[Any]]: The result of calculating the energy volume curve.

    """
    input_parameter_lst, output_as_lst = combine_structure_and_potential(
        structure=structure, potential_dataframe=potential_dataframe
    )
    enable_mpi, lmp = _check_mpi(executor=executor, lmp=lmp)
    input_parameter_dict = [
        {
            "lmp": lmp,
            "enable_mpi": enable_mpi,
            "structure": struct,
            "potential_dataframe": pot,
            "num_points": num_points,
            "fit_type": fit_type,
            "fit_order": fit_order,
            "vol_range": vol_range,
            "axes": axes,
            "strains": strains,
            "minimization_activated": minimization_activated,
        }
        for [struct, pot] in input_parameter_lst
    ]
    if executor is None:
        executor = InProcessExecutor()
    result_lst = [
        fut.result()
        for fut in [
            executor.submit(calculate_energy_volume_curve, **kwargs)
            for kwargs in input_parameter_dict
        ]
    ]
    if output_as_lst:
        return result_lst
    else:
        return result_lst[0]


def combine_structure_and_potential(
    structure: Union[List[Atoms], np.ndarray],
    potential_dataframe: Union[List[DataFrame], DataFrame, Series],
) -> Tuple[List[List[Any]], bool]:
    """
    Combine the structure and potential dataframe into a list of input parameters.

    Args:
        structure (Union[List[Atoms], np.ndarray]): The structure to be combined.
        potential_dataframe (Union[List[DataFrame], DataFrame, Series]): The potential dataframe to be combined.

    Returns:
        Tuple[List[List[Any]], bool]: The combined input parameters and a flag indicating if the output should be a list.

    Raises:
        ValueError: If the length of structure and potential_dataframe is not equal.
        TypeError: If the types of structure and potential_dataframe are not valid.

    """
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
                "potential_dataframe should either be a pandas.DataFrame object or a list of those. "
            )
    elif isinstance(structure, Atoms):
        if isinstance(potential_dataframe, (list, np.ndarray)):
            return [[structure, p] for p in potential_dataframe], True
        elif isinstance(potential_dataframe, (DataFrame, Series)):
            return [[structure, potential_dataframe]], False
        else:
            raise TypeError(
                "potential_dataframe should either be a pandas.DataFrame object or a list of those. "
            )
    else:
        raise TypeError(
            "structure should either be an ase.atoms.Atoms object or a list of those."
        )
