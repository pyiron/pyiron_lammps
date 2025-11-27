import unittest
import os
import shutil
import numpy as np
import pandas
from ase.atoms import Atoms
import ase.units as units

from pyiron_lammps.compatibility.file import lammps_file_interface_function


class TestCompatibilityFile(unittest.TestCase):
    def setUp(self):
        self.working_dir = os.path.abspath(os.path.join(__file__, "..", "lmp"))
        self.static_path = os.path.abspath(
            os.path.join("..", os.path.dirname(__file__), "static")
        )
        self.keys = [
            "steps",
            "natoms",
            "cells",
            "indices",
            "forces",
            "velocities",
            "unwrapped_positions",
            "positions",
            "temperature",
            "energy_pot",
            "energy_tot",
            "volume",
            "pressures",
        ]

    def tearDown(self):
        if os.path.exists(self.working_dir):
            shutil.rmtree(self.working_dir)

    def test_water_calculation(self):
        density = 1.0e-24  # g/A^3
        n_mols = 27
        mol_mass_water = 18.015  # g/mol

        # Determining the supercell size size
        mass = mol_mass_water * n_mols / units.mol  # g
        vol_h2o = mass / density  # in A^3
        a = vol_h2o ** (1.0 / 3.0)  # A

        # Constructing the unitcell
        n = int(round(n_mols ** (1.0 / 3.0)))

        dx = 0.7
        r_O = [0, 0, 0]
        r_H1 = [dx, dx, 0]
        r_H2 = [-dx, dx, 0]
        unit_cell = (a / n) * np.eye(3)
        water_potential = pandas.DataFrame(
            {
                "Name": ["H2O_tip3p"],
                "Filename": [[]],
                "Model": ["TIP3P"],
                "Species": [["H", "O"]],
                "Config": [
                    [
                        "# @potential_species H_O ### species in potential\n",
                        "# W.L. Jorgensen et.al., The Journal of Chemical Physics 79, 926 (1983); https://doi.org/10.1063/1.445869\n",
                        "#\n",
                        "\n",
                        "units real\n",
                        "dimension 3\n",
                        "atom_style full\n",
                        "\n",
                        "# create groups ###\n",
                        "group O type 2\n",
                        "group H type 1\n",
                        "\n",
                        "## set charges - beside manually ###\n",
                        "set group O charge -0.830\n",
                        "set group H charge 0.415\n",
                        "\n",
                        "### TIP3P Potential Parameters ###\n",
                        "pair_style lj/cut/coul/long 10.0\n",
                        "pair_coeff * * 0.0 0.0 \n",
                        "pair_coeff 2 2 0.102 3.188 \n",
                        "bond_style harmonic\n",
                        "bond_coeff 1 450 0.9572\n",
                        "angle_style harmonic\n",
                        "angle_coeff 1 55 104.52\n",
                        "kspace_style pppm 1.0e-5\n",
                        "\n",
                    ]
                ],
            }
        )
        water_base = Atoms(
            symbols=["H", "H", "O"],
            positions=[r_H1, r_H2, r_O],
            cell=unit_cell,
            pbc=True,
        )
        water = water_base.repeat([n, n, n])
        _, parsed_output, job_crashed = lammps_file_interface_function(
            working_directory="lmp",
            structure=water,
            potential=water_potential,
            calc_mode="md",
            calc_kwargs={"temperature": 300, "n_ionic_steps": 1e4, "time_step": 0.01},
            units="real",
            lmp_command="cp "
            + str(os.path.join(self.static_path, "compatibility_output"))
            + "/* .",
            resource_path=None,
        )
        self.assertFalse(job_crashed)
        for key in self.keys:
            self.assertIn(key, parsed_output["generic"])
