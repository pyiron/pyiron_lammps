from concurrent.futures import Executor, Future
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from ase.atoms import Atoms
from atomistics.calculators import evaluate_with_lammps_library
from atomistics.workflows import (
    ElasticMatrixWorkflow,
    EnergyVolumeCurveWorkflow,
    optimize_positions_and_volume,
)
from pandas import DataFrame, Series
from pylammpsmpi import LammpsASELibrary


class InProcessExecutor:
    @staticmethod
    def submit(funct: Callable, *args, **kwargs):
        f = Future()
        f.set_result(funct(*args, **kwargs))
        return f


def _get_lmp(
    lmp: Optional[LammpsASELibrary] = None, enable_mpi: bool = False
) -> Tuple[LammpsASELibrary, bool]:
    # Create temporary LAMMPS instance if necessary
    if lmp is None:
        close_lmp_after_calculation = True
        if enable_mpi:
            # To get the right instance of MPI.COMM_SELF it is necessary to import it inside the function.
            from mpi4py import MPI

            lmp = LammpsASELibrary(
                working_directory=None,
                cores=1,
                comm=MPI.COMM_SELF,
                logger=None,
                log_file=None,
                library=None,
                disable_log_file=True,
            )
        else:
            lmp = LammpsASELibrary(
                working_directory=None,
                cores=1,
                comm=None,
                logger=None,
                log_file=None,
                library=None,
                disable_log_file=True,
            )
    else:
        close_lmp_after_calculation = False
    return lmp, close_lmp_after_calculation


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


def _optimize_structure_optional(
    lmp: Optional[LammpsASELibrary],
    structure: Atoms,
    potential_dataframe: DataFrame,
    minimization_activated=True,
):
    """
    Optimize the structure using LAMMPS if minimization is activated, otherwise return the original structure.

    Args:
        lmp (LammpsLibrary): The LAMMPS library object.
        structure (Structure): The structure to be optimized.
        potential_dataframe (pandas.DataFrame): The potential dataframe.
        minimization_activated (bool, optional): Flag to activate minimization. Defaults to True.

    Returns:
        Structure: The optimized structure if minimization is activated, otherwise the original structure.
    """
    if minimization_activated:
        return optimize_structure(
            lmp=lmp, structure=structure, potential_dataframe=potential_dataframe
        )
    else:
        return structure


def optimize_structure(
    lmp: Optional[LammpsASELibrary],
    structure: Atoms,
    potential_dataframe: DataFrame,
    enable_mpi: bool = False,
):
    """
    Optimize the structure by optimizing positions and volume using LAMMPS.

    Args:
        lmp (LammpsLibrary): The LAMMPS library object.
        structure (Structure): The structure to be optimized.
        potential_dataframe (pandas.DataFrame): The potential dataframe.
        enable_mpi (bool): Flag to enable MPI. Default is False.

    Returns:
        Structure: The optimized structure with optimized positions and volume.
    """
    lmp, close_lmp_after_calculation = _get_lmp(lmp=lmp, enable_mpi=enable_mpi)
    task_dict = optimize_positions_and_volume(structure=structure)
    structure_copy = evaluate_with_lammps_library(
        task_dict=task_dict,
        potential_dataframe=potential_dataframe,
        lmp=lmp,
        lmp_optimizer_kwargs={},
    )["structure_with_optimized_positions_and_volume"]

    # clean memory
    lmp.interactive_lib_command("clear")
    if close_lmp_after_calculation:
        lmp.close()
    return structure_copy


def calculate_elastic_constants(
    lmp: Optional[LammpsASELibrary],
    structure: Atoms,
    potential_dataframe: DataFrame,
    num_of_point: int = 5,
    eps_range: float = 0.005,
    sqrt_eta: bool = True,
    fit_order: int = 2,
    minimization_activated: bool = False,
    enable_mpi: bool = False,
) -> np.ndarray:
    """
    Calculate the elastic constants of a structure using LAMMPS.

    Args:
        lmp (LammpsLibrary): The LAMMPS library object.
        structure (Structure): The structure to calculate the elastic constants for.
        potential_dataframe (pandas.DataFrame): The potential dataframe.
        num_of_point (int, optional): The number of strain points to use. Defaults to 5.
        eps_range (float, optional): The range of strain to use. Defaults to 0.005.
        sqrt_eta (bool, optional): Flag to use square root of eta. Defaults to True.
        fit_order (int, optional): The order of the polynomial fit. Defaults to 2.
        minimization_activated (bool, optional): Flag to activate minimization. Defaults to False.
        enable_mpi (bool): Flag to enable MPI. Default is False.

    Returns:
        np.ndarray: The elastic constants matrix.
    """
    lmp, close_lmp_after_calculation = _get_lmp(lmp=lmp, enable_mpi=enable_mpi)

    # Optimize structure
    structure_opt = _optimize_structure_optional(
        lmp=lmp,
        structure=structure,
        potential_dataframe=potential_dataframe,
        minimization_activated=minimization_activated,
    )

    # Generate structures
    calculator = ElasticMatrixWorkflow(
        structure=structure_opt.copy(),
        num_of_point=num_of_point,
        eps_range=eps_range,
        sqrt_eta=sqrt_eta,
        fit_order=fit_order,
    )
    structure_dict = calculator.generate_structures()

    # run calculation
    energy_tot_lst = evaluate_with_lammps_library(
        task_dict=structure_dict,
        potential_dataframe=potential_dataframe,
        lmp=lmp,
        lmp_optimizer_kwargs={},
    )

    # fit
    result_dict = calculator.analyse_structures(
        output_dict=energy_tot_lst,
        output_keys=("elastic_matrix",),
    )

    if close_lmp_after_calculation:
        lmp.close()
    return result_dict["elastic_matrix"]


def calculate_energy_volume_curve(
    lmp: Optional[LammpsASELibrary],
    structure: Atoms,
    potential_dataframe: DataFrame,
    num_points: int = 11,
    fit_type: str = "polynomial",
    fit_order: int = 3,
    vol_range: float = 0.05,
    axes: Tuple[str, str, str] = ("x", "y", "z"),
    strains: Optional[List[float]] = None,
    minimization_activated: bool = False,
    enable_mpi: bool = False,
) -> Dict[str, Any]:
    """
    Calculate the energy-volume curve of a structure using LAMMPS.

    Args:
        lmp (LammpsLibrary): The LAMMPS library object.
        structure (Structure): The structure to calculate the energy-volume curve for.
        potential_dataframe (pandas.DataFrame): The potential dataframe.
        num_points (int, optional): The number of volume points to use. Defaults to 11.
        fit_type (str, optional): The type of fit to use. Defaults to "polynomial".
        fit_order (int, optional): The order of the fit. Defaults to 3.
        vol_range (float, optional): The range of volume to use. Defaults to 0.05.
        axes (Tuple[str, str, str], optional): The axes to vary the volume along. Defaults to ("x", "y", "z").
        strains (List[float], optional): The strains to apply to the structure. Defaults to None.
        minimization_activated (bool, optional): Flag to activate minimization. Defaults to False.
        enable_mpi (bool): Flag to enable MPI. Default is False.

    Returns:
        Dict[str, Any]: The fit results.
    """
    lmp, close_lmp_after_calculation = _get_lmp(lmp=lmp, enable_mpi=enable_mpi)

    # Optimize structure
    structure_opt = _optimize_structure_optional(
        lmp=lmp,
        structure=structure,
        potential_dataframe=potential_dataframe,
        minimization_activated=minimization_activated,
    )

    # Generate structures
    calculator = EnergyVolumeCurveWorkflow(
        structure=structure_opt.copy(),
        num_points=num_points,
        fit_type=fit_type,
        fit_order=fit_order,
        vol_range=vol_range,
        axes=axes,
        strains=strains,
    )
    structure_dict = calculator.generate_structures()

    # run calculation
    energy_tot_lst = evaluate_with_lammps_library(
        task_dict=structure_dict,
        potential_dataframe=potential_dataframe,
        lmp=lmp,
    )

    # fit
    calculator.analyse_structures(energy_tot_lst)

    if close_lmp_after_calculation:
        lmp.close()

    return calculator.fit_dict


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
