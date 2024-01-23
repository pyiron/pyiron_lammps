from pyiron_lammps.decorator import calculation
from atomistics.workflows import (
    ElasticMatrixWorkflow,
    EnergyVolumeCurveWorkflow,
    optimize_positions_and_volume,
)
from atomistics.calculators import evaluate_with_lammps_library


def _optimize_structure_optional(
    lmp, structure, potential_dataframe, minimization_activated=True
):
    if minimization_activated:
        return optimize_structure(
            lmp=lmp, structure=structure, potential_dataframe=potential_dataframe
        )
    else:
        return structure


@calculation
def optimize_structure(lmp, structure, potential_dataframe):
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
    num_of_point=5,
    eps_range=0.005,
    sqrt_eta=True,
    fit_order=2,
    minimization_activated=False,
):
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
    num_points=11,
    fit_type="polynomial",
    fit_order=3,
    vol_range=0.05,
    axes=("x", "y", "z"),
    strains=None,
    minimization_activated=False,
):
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
