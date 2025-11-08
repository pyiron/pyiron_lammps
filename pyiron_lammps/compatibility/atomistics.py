import os

from pyiron_lammps import DUMP_COMMANDS
from pyiron_lammps.structure import write_lammps_datafile


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


def write_lammps_input_file(
    working_directory, structure, potential_dataframe, input_template, units="metal"
):
    write_lammps_datafile(
        structure=structure,
        potential_elements=potential_dataframe["Species"],
        bond_dict=None,
        units=units,
        file_name="lammps.data",
        working_directory=working_directory,
    )
    input_str = (
        "\n".join(lammps_file_initialization(structure=structure, units=units))
        + "\n".join(potential_dataframe["Config"])
        + "\n"
        + "".join(DUMP_COMMANDS)
        + input_template
    )
    with open(os.path.join(working_directory, "lmp.in"), "w") as f:
        f.writelines(input_str)
