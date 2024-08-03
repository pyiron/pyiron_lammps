from concurrent.futures import Executor
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


def _get_lammps_mpi(enable_mpi: bool = True) -> LammpsASELibrary:
    """
    Get an instance of LammpsASELibrary for parallel execution using MPI.

    Args:
        enable_mpi (bool): Flag to enable MPI. Default is True.

    Returns:
        LammpsASELibrary: An instance of LammpsASELibrary.

    """
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


def _parallel_execution(
    function: Callable[..., Any],
    input_parameter_lst: List[List[Any]],
    lmp: Optional[LammpsASELibrary] = None,
    executor: Optional[Executor] = None
) -> List[Any]:
    """
    Execute a function in parallel using either a provided executor or the default executor.

    Args:
        function (Callable): The function to be executed in parallel.
        input_parameter_lst (List[List[Any]]): A list of input parameters for each function call.
        lmp (Optional[LammpsASELibrary]): An optional instance of LammpsASELibrary for serial execution.
        executor (Optional[Executor]): An optional executor for parallel execution.

    Returns:
        List[Any]: A list of results from each function call.

    Raises:
        ValueError: If the external LAMMPS instance is provided for parallel execution.
        ValueError: If the number of cores is not a positive integer.
    """
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


def _optimize_structure_serial(input_parameter: Tuple[Atoms, Union[List[DataFrame], DataFrame], bool, Optional[LammpsASELibrary]]) -> Any:
    """
    Optimize the structure using serial execution.

    Args:
        input_parameter (Tuple[Atoms, Union[List[DataFrame], DataFrame], bool, Optional[LammpsASELibrary]]): The input parameters for optimization.

    Returns:
        Any: The result of the optimization.

    """
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


def _calculate_elastic_constants_serial(input_parameter: Tuple[Atoms, Union[List[DataFrame], DataFrame], int, float, bool, int, bool, bool, Optional[LammpsASELibrary]]) -> Any:
    """
    Calculate the elastic constants using serial execution.

    Args:
        input_parameter (Tuple[Atoms, Union[List[DataFrame], DataFrame], int, float, bool, int, bool, bool, Optional[LammpsASELibrary]]): The input parameters for calculating elastic constants.

    Returns:
        Any: The result of calculating elastic constants.

    """
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


def _calculate_energy_volume_curve_serial(input_parameter: Tuple[Atoms, Union[List[DataFrame], DataFrame], int, str, int, float, Tuple[str, str, str], Optional[List[float]], bool, bool, Optional[LammpsASELibrary]]) -> Any:
    """
    Calculate the energy volume curve using serial execution.

    Args:
        input_parameter (Tuple[Atoms, Union[List[DataFrame], DataFrame], int, str, int, float, Tuple[str, str, str], Optional[List[float]], bool, bool, Optional[LammpsASELibrary]]): The input parameters for calculating the energy volume curve.

    Returns:
        Any: The result of calculating the energy volume curve.

    """
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
    structure: Any,
    potential_dataframe: Union[List[DataFrame], DataFrame],
    lmp: Optional[LammpsASELibrary] = None,
    executor: Optional[Executor] = None
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


def combine_structure_and_potential(structure: Union[List[Atoms], np.ndarray], potential_dataframe: Union[List[DataFrame], DataFrame, Series]) -> Tuple[List[List[Any]], bool]:
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
