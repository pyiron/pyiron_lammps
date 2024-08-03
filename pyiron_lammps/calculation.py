from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from atomistics.calculators import evaluate_with_lammps_library
from atomistics.workflows import (
    ElasticMatrixWorkflow,
    EnergyVolumeCurveWorkflow,
    optimize_positions_and_volume,
)

from pyiron_lammps.decorator import calculation


def _optimize_structure_optional(
    lmp, structure, potential_dataframe, minimization_activated=True
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


@calculation
def optimize_structure(lmp, structure, potential_dataframe):
    """
    Optimize the structure by optimizing positions and volume using LAMMPS.

    Args:
        lmp (LammpsLibrary): The LAMMPS library object.
        structure (Structure): The structure to be optimized.
        potential_dataframe (pandas.DataFrame): The potential dataframe.

    Returns:
        Structure: The optimized structure with optimized positions and volume.
    """
    task_dict = optimize_positions_and_volume(structure=structure)
    structure_copy = evaluate_with_lammps_library(
        task_dict=task_dict,
        potential_dataframe=potential_dataframe,
        lmp=lmp,
        lmp_optimizer_kwargs={},
    )["structure_with_optimized_positions_and_volume"]

    # clean memory
    lmp.interactive_lib_command("clear")
    return structure_copy


@calculation
def calculate_elastic_constants(
    lmp,
    structure,
    potential_dataframe,
    num_of_point: int = 5,
    eps_range: float = 0.005,
    sqrt_eta: bool = True,
    fit_order: int = 2,
    minimization_activated: bool = False,
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

    Returns:
        np.ndarray: The elastic constants matrix.
    """
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
    return result_dict["elastic_matrix"]


@calculation
def calculate_energy_volume_curve(
    lmp,
    structure,
    potential_dataframe,
    num_points: int = 11,
    fit_type: str = "polynomial",
    fit_order: int = 3,
    vol_range: float = 0.05,
    axes: Tuple[str, str, str] = ("x", "y", "z"),
    strains: Optional[List[float]] = None,
    minimization_activated: bool = False,
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

    Returns:
        Dict[str, Any]: The fit results.
    """
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
    return calculator.fit_dict
