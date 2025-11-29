import unittest
import os
import shutil
import numpy as np
import pandas
from ase.atoms import Atoms
import ase.units as units

from pyiron_lammps.compatibility.file import lammps_file_interface_function
from pyiron_lammps.compatibility.structure import (
    LammpsStructureCompatibility,
    write_lammps_datafile,
    get_charge,
    _find_line_by_prefix,
)


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


class TestLammpsStructureCompatibility(unittest.TestCase):
    def setUp(self):
        self.output_folder = os.path.abspath(
            os.path.join(__file__, "..", "structure_comp")
        )
        os.makedirs(self.output_folder, exist_ok=True)

    def tearDown(self):
        if os.path.exists(self.output_folder):
            shutil.rmtree(self.output_folder)

    def test_structure_bond_3d(self):
        ls = LammpsStructureCompatibility(atom_type="bond")
        ls.el_eam_lst = ["H"]
        atoms = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]], cell=np.eye(3) * 15)
        ls.structure = atoms
        output_lines = ls._string_input.split("\n")
        atoms_part_found = False
        bonds_part_found = False
        for i, line in enumerate(output_lines):
            if line.strip() == "Atoms":
                atoms_part_found = True
                self.assertEqual(
                    output_lines[i + 2].strip(), "1 1 1 0.000000 0.000000 0.000000"
                )
                self.assertEqual(
                    output_lines[i + 3].strip(), "2 1 1 0.740000 0.000000 0.000000"
                )
            if line.strip() == "Bonds":
                bonds_part_found = True
                self.assertEqual(output_lines[i + 2].strip(), "1 1 1 2")

        self.assertTrue(atoms_part_found)
        self.assertTrue(bonds_part_found)

    def test_structure_bond_2d(self):
        ls = LammpsStructureCompatibility(atom_type="bond")
        ls.el_eam_lst = ["H"]
        atoms = Atoms(
            "H2",
            positions=[[0, 0, 0], [0.74, 0, 0]],
            cell=[[15, 15, 0], [15, -15, 0], [0, 0, 15]],
        )
        atoms.pbc = True
        ls.structure = atoms

        output_lines = ls._string_input.split("\n")
        atoms_part_found = False
        for i, line in enumerate(output_lines):
            if line.strip() == "Atoms":
                atoms_part_found = True
                self.assertIn("1 1 1 0.000000 0.000000 0.000000", output_lines[i + 2])
                self.assertIn("2 1 1 0.523259 0.523259 0.000000", output_lines[i + 3])
        self.assertTrue(atoms_part_found)

    def test_structure_bond_1d_error(self):
        ls = LammpsStructureCompatibility(atom_type="bond")
        ls.el_eam_lst = ["H"]
        atoms = Atoms("H", positions=[[0, 0, 0]], cell=[15, 15, 15])
        ls.structure = atoms
        self.assertNotIn("Bonds", ls._string_input)

    def test_structure_full(self):
        bond_dict = {
            "O": {
                "element_list": ["H"],
                "cutoff_list": [1.2],
                "max_bond_list": [2],
                "bond_type_list": [1],
                "angle_type_list": [1],
            }
        }
        q_dict = {"H": 0.41, "O": -0.82}
        ls = LammpsStructureCompatibility(
            atom_type="full", bond_dict=bond_dict, q_dict=q_dict
        )
        ls.el_eam_lst = ["H", "O"]

        atoms = Atoms(
            "H2O",
            positions=[(0, 0.757, 0.586), (0, -0.757, 0.586), (0, 0, 0)],
            cell=np.eye(3) * 15,
        )
        ls.structure = atoms
        output_lines = ls._string_input.split("\n")

        atoms_part_found = False
        bonds_part_found = False
        angles_part_found = False

        for i, line in enumerate(output_lines):
            if line.strip() == "Atoms":
                atoms_part_found = True
                self.assertIn("1 1 1 0.410000", output_lines[i + 2])
                self.assertIn("2 1 1 0.410000", output_lines[i + 3])
                self.assertIn("3 1 2 -0.820000", output_lines[i + 4])
            if line.strip() == "Bonds":
                bonds_part_found = True
                line1 = output_lines[i + 2].strip()
                line2 = output_lines[i + 3].strip()
                self.assertTrue(
                    ("1 1 3 1" == line1 and "2 1 3 2" == line2)
                    or ("1 1 3 2" == line1 and "2 1 3 1" == line2)
                )

            if line.strip() == "Angles":
                angles_part_found = True
                self.assertTrue(
                    "1 1 1 3 2" in output_lines[i + 2]
                    or "1 1 2 3 1" in output_lines[i + 2]
                )

        self.assertTrue(atoms_part_found)
        self.assertTrue(bonds_part_found)
        self.assertTrue(angles_part_found)

    def test_structure_full_no_angles(self):
        bond_dict = {
            "O": {
                "element_list": ["H"],
                "cutoff_list": [1.2],
                "max_bond_list": [2],
                "bond_type_list": [1],
                "angle_type_list": [None],
            }
        }
        q_dict = {"H": 0.41, "O": -0.82}
        ls = LammpsStructureCompatibility(
            atom_type="full", bond_dict=bond_dict, q_dict=q_dict
        )
        ls.el_eam_lst = ["H", "O"]

        atoms = Atoms(
            "H2O",
            positions=[(0, 0.757, 0.586), (0, -0.757, 0.586), (0, 0, 0)],
            cell=np.eye(3) * 15,
        )
        ls.structure = atoms

        self.assertNotIn("Angles", ls._string_input)

    def test_structure_full_no_bonds_for_element(self):
        bond_dict = {
            "C": {
                "element_list": ["H"],
                "cutoff_list": [1.2],
                "max_bond_list": [1],
                "bond_type_list": [1],
                "angle_type_list": [1],
            }
        }
        q_dict = {"H": 0.41, "O": -0.82}
        ls = LammpsStructureCompatibility(
            atom_type="full", bond_dict=bond_dict, q_dict=q_dict
        )
        ls.el_eam_lst = ["H", "O"]

        atoms = Atoms(
            "H2O",
            positions=[(0, 0.757, 0.586), (0, -0.757, 0.586), (0, 0, 0)],
            cell=np.eye(3) * 15,
        )
        ls.structure = atoms

        self.assertNotIn("Bonds", ls._string_input)
        self.assertNotIn("Angles", ls._string_input)

    def test_write_lammps_datafile_full(self):
        atoms = Atoms(
            "H2O",
            positions=[(0, 0.757, 0.586), (0, -0.757, 0.586), (0, 0, 0)],
            cell=np.eye(3) * 15,
        )
        potential_lst = ["set group H charge 0.41", "set group O charge -0.82"]

        write_lammps_datafile(
            structure=atoms,
            potential_elements=["H", "O"],
            file_name="lammps.data.full",
            working_directory=self.output_folder,
            atom_type="full",
            potential_lst=potential_lst,
        )

        with open(os.path.join(self.output_folder, "lammps.data.full"), "r") as f:
            content = f.read()

        self.assertIn("Atoms", content)
        self.assertIn("Bonds", content)
        self.assertIn("Angles", content)
        self.assertIn("1 1 1 0.410000", content)
        self.assertIn("2 1 1 0.410000", content)
        self.assertIn("3 1 2 -0.820000", content)
        self.assertTrue(
            ("1 1 3 1" in content and "2 1 3 2" in content)
            or ("1 1 3 2" in content and "2 1 3 1" in content)
        )
        self.assertTrue("1 1 1 3 2" in content or "1 1 2 3 1" in content)

    def test_get_charge(self):
        potential_lines = [
            "set group Fe charge 2.0",
            "set group O charge -2.0",
            "  set group Si charge   4.0 # comment",
        ]
        self.assertEqual(get_charge(potential_lines, "Fe"), 2.0)
        self.assertEqual(get_charge(potential_lines, "O"), -2.0)
        self.assertEqual(get_charge(potential_lines, "Si"), 4.0)
        with self.assertRaises(NameError):
            get_charge(potential_lines, "C")

    def test_find_line_by_prefix(self):
        lines = ["some line", "prefix then something", "another line"]
        self.assertEqual(
            _find_line_by_prefix(lines, "prefix"), ["prefix", "then", "something"]
        )
        with self.assertRaises(ValueError):
            _find_line_by_prefix(lines, "nonexistent")

    def test_structure_setter_charge(self):
        ls = LammpsStructureCompatibility(atom_type="charge")
        ls.el_eam_lst = ["Fe"]
        atoms = Atoms("Fe", positions=[[0, 0, 0]], cell=np.eye(3))
        atoms.set_initial_charges([1.5])
        ls.structure = atoms
        self.assertIn("1 1 1.500000 0.000000", ls._string_input)
