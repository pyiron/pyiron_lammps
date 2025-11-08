import os
import subprocess
from typing import Optional

from ase.atoms import Atoms

from pyiron_lammps.compatibility.calculate import (
    calc_md,
    calc_minimize,
    calc_static,
)
from pyiron_lammps.output import parse_lammps_output
from pyiron_lammps.potential import get_potential_by_name
from pyiron_lammps.structure import write_lammps_datafile


def lammps_file_interface_function(
    working_directory: str,
    structure: Atoms,
    potential: str,
    calc_mode: str = "static",
    calc_kwargs: Optional[dict] = None,
    units: str = "metal",
    lmp_command: str = "mpiexec -n 1 --oversubscribe lmp_mpi -in lmp.in",
    resource_path: Optional[str] = None,
):
    """
    A single function to execute a LAMMPS calculation based on the LAMMPS job implemented in pyiron

    Examples:

    >>> import os
    >>> from ase.build import bulk
    >>> from pyiron_atomistics.lammps.lammps import lammps_function
    >>>
    >>> shell_output, parsed_output, job_crashed = lammps_function(
    ...     working_directory=os.path.abspath("lmp_working_directory"),
    ...     structure=bulk("Al", cubic=True),
    ...     potential='2009--Mendelev-M-I--Al-Mg--LAMMPS--ipr1',
    ...     calc_mode="md",
    ...     calc_kwargs={"temperature": 500.0, "pressure": 0.0, "n_ionic_steps": 1000, "n_print": 100},
    ...     cutoff_radius=None,
    ...     units="metal",
    ...     bonds_kwargs={},
    ...      enable_h5md=False,
    ... )

    Args:
        working_directory (str): directory in which the LAMMPS calculation is executed
        structure (Atoms): ase.atoms.Atoms - atomistic structure
        potential (str): Name of the LAMMPS potential based on the NIST database and the OpenKIM database
        calc_mode (str): select mode of calculation ["static", "md", "minimize", "vcsgc"]
        calc_kwargs (dict): key-word arguments for the calculate function, the input parameters depend on the calc_mode:
          "static": No parameters
          "md": "temperature", "pressure", "n_ionic_steps", "time_step", "n_print", "temperature_damping_timescale",
                "pressure_damping_timescale", "seed", "tloop", "initial_temperature", "langevin", "delta_temp",
                "delta_press", job_name", "rotation_matrix"
          "minimize": "ionic_energy_tolerance", "ionic_force_tolerance", "max_iter", "pressure", "n_print", "style",
                      "rotation_matrix"
          "vcsgc": "mu", "ordered_element_list", "target_concentration", "kappa", "mc_step_interval", "swap_fraction",
                   "temperature_mc", "window_size", "window_moves", "temperature", "pressure", "n_ionic_steps",
                   "time_step", "n_print", "temperature_damping_timescale", "pressure_damping_timescale", "seed",
                   "initial_temperature", "langevin", "job_name", "rotation_matrix"
        cutoff_radius (float): cut-off radius for the interatomic potential
        units (str): Units for LAMMPS
        bonds_kwargs (dict): key-word arguments to create atomistic bonds:
          "species", "element_list", "cutoff_list", "max_bond_list", "bond_type_list", "angle_type_list",
        server_kwargs (dict): key-word arguments to create server object - the available parameters are:
          "user", "host", "run_mode", "queue", "qid", "cores", "threads", "new_h5", "structure_id", "run_time",
          "memory_limit", "accept_crash", "additional_arguments", "gpus", "conda_environment_name",
          "conda_environment_path"
        enable_h5md (bool): activate h5md mode for LAMMPS
        write_restart_file (bool): enable writing the LAMMPS restart file
        read_restart_file (bool): enable loading the LAMMPS restart file
        restart_file (str): file name of the LAMMPS restart file to copy
        executable_version (str): LAMMPS version to for the execution
        executable_path (str): path to the LAMMPS executable
        input_control_file (str|list|dict): Option to modify the LAMMPS input file directly

    Returns:
        str, dict, bool: Tuple consisting of the shell output (str), the parsed output (dict) and a boolean flag if
                         the execution raised an accepted error.
    """
    if calc_kwargs is None:
        calc_kwargs = {}

    os.makedirs(working_directory, exist_ok=True)
    potential_dataframe = get_potential_by_name(
        potential_name=potential, resource_path=resource_path
    )
    write_lammps_datafile(
        structure=structure,
        potential_elements=potential_dataframe["Species"],
        bond_dict=None,
        units=units,
        file_name="lammps.data",
        working_directory=working_directory,
    )
    lmp_str_lst = lammps_file_initialization(structure=structure)
    lmp_str_lst += potential_dataframe["Config"]
    lmp_str_lst += ["variable dumptime equal {} ".format(calc_kwargs.get("n_print", 1))]
    lmp_str_lst += [
        "dump 1 all custom ${dumptime} dump.out id type xsu ysu zsu fx fy fz vx vy vz",
        'dump_modify 1 sort id format line "%d %d %20.15g %20.15g %20.15g %20.15g %20.15g %20.15g %20.15g %20.15g %20.15g"',
    ]

    if calc_mode == "static":
        lmp_str_lst += calc_static()
    elif calc_mode == "md":
        if "n_ionic_steps" in calc_kwargs.keys():
            n_ionic_steps = calc_kwargs.pop("n_ionic_steps")
        else:
            n_ionic_steps = 1
        lmp_str_lst += calc_md(**calc_kwargs)
        lmp_str_lst += ["run {} ".format(n_ionic_steps)]
    elif calc_mode == "minimize":
        lmp_str_lst += calc_minimize(**calc_kwargs)
    else:
        raise ValueError(
            f"calc_mode must be one of: static, md or minimize, not {calc_mode}"
        )

    with open(os.path.join(working_directory, "lmp.in"), "w") as f:
        f.writelines([l + "\n" for l in lmp_str_lst])

    shell = subprocess.check_output(
        lmp_command,
        cwd=working_directory,
        shell=True,
        universal_newlines=True,
        env=os.environ.copy(),
    )
    output = parse_lammps_output(
        working_directory=working_directory,
        structure=structure,
        potential_elements=potential_dataframe["Species"],
        units=units,
        prism=None,
        dump_h5_file_name="dump.h5",
        dump_out_file_name="dump.out",
        log_lammps_file_name="log.lammps",
    )
    return shell, output, False


def lammps_file_initialization(structure, dimension=3, units="metal"):
    boundary = " ".join(["p" if coord else "f" for coord in structure.pbc])
    init_commands = [
        "units " + units,
        "dimension " + str(dimension),
        "boundary " + boundary + "",
        "atom_style atomic",
        "read_data lammps.data",
    ]
    return init_commands
