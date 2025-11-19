import unittest
import shutil
import os
from ase.build import bulk
import numpy as np

from pyiron_lammps.potential import get_potential_by_name
from pyiron_lammps.compatibility.file import lammps_file_interface_function


class TestCompatibilityFile(unittest.TestCase):
    def setUp(self):
        self.structure = bulk("Al", cubic=True).repeat([2, 2, 2])
        self.potential = get_potential_by_name(
            potential_name="2009--Mendelev-M-I--Al-Mg--LAMMPS--ipr1",
            resource_path=os.path.abspath(
                os.path.join("..", os.path.dirname(__file__), "static", "lammps")
            ),
        )
        self.units = "metal"
        self.working_directory = os.path.abspath("lmp")
        self.output_key_lst = [
            "steps",
            "natoms",
            "cells",
            "indices",
            "forces",
            "velocities",
            "positions",
            "temperature",
            "energy_pot",
            "energy_tot",
            "volume",
            "pressures",
        ]
        self.output_key_1d_lst = [
            "steps",
            "natoms",
            "temperature",
            "energy_pot",
            "energy_tot",
            "volume",
        ]
        self.output_key_3d_atoms_lst = ["forces", "velocities", "positions"]
        self.output_key_1d_atoms_lst = ["indices"]
        self.output_key_9_lst = ["cells", "pressures"]

    def tearDown(self):
        if os.path.exists(self.working_directory):
            shutil.rmtree(self.working_directory)

    def test_calc_static(self):
        shell_output, parsed_output, job_crashed = lammps_file_interface_function(
            working_directory=self.working_directory,
            structure=self.structure,
            potential=self.potential,
            calc_mode="static",
            units=self.units,
        )
        for k in self.output_key_lst:
            self.assertTrue(k in parsed_output.keys())

    def test_minimize_positions(self):
        shell_output, parsed_output, job_crashed = lammps_file_interface_function(
            working_directory=self.working_directory,
            structure=self.structure,
            potential=self.potential,
            calc_mode="minimize",
            units=self.units,
        )
        for k in self.output_key_lst:
            self.assertTrue(k in parsed_output.keys())

    def test_minimize_volume(self):
        calc_kwargs = {"pressure": 0.0}
        shell_output, parsed_output, job_crashed = lammps_file_interface_function(
            working_directory=self.working_directory,
            structure=self.structure,
            potential=self.potential,
            calc_mode="minimize",
            calc_kwargs=calc_kwargs,
            units=self.units,
        )
        for k in self.output_key_lst:
            self.assertTrue(k in parsed_output.keys())

    def test_calc_md_nose_hoover_nvt(self):
        calc_kwargs = {"temperature": 500.0, "n_ionic_steps": 1000, "n_print": 100}
        shell_output, parsed_output, job_crashed = lammps_file_interface_function(
            working_directory=self.working_directory,
            structure=self.structure,
            potential=self.potential,
            calc_mode="md",
            calc_kwargs=calc_kwargs,
            units=self.units,
        )
        for k in self.output_key_lst:
            self.assertTrue(k in parsed_output.keys())
            self.assertEqual(len(parsed_output[k]), 10)

        for k in self.output_key_3d_atoms_lst:
            self.assertEqual(np.array(parsed_output[k]).shape, (10, 32, 3))

        for k in self.output_key_1d_atoms_lst:
            self.assertEqual(np.array(parsed_output[k]).shape, (10, 32))

        for k in self.output_key_9_lst:
            self.assertEqual(np.array(parsed_output[k]).shape, (10, 3, 3))

    def test_calc_md_nose_hoover_npt(self):
        calc_kwargs = {
            "temperature": 500.0,
            "pressure": 0.0,
            "n_ionic_steps": 1000,
            "n_print": 100,
        }
        shell_output, parsed_output, job_crashed = lammps_file_interface_function(
            working_directory=self.working_directory,
            structure=self.structure,
            potential=self.potential,
            calc_mode="md",
            calc_kwargs=calc_kwargs,
            units=self.units,
        )
        for k in self.output_key_lst:
            self.assertTrue(k in parsed_output.keys())
            self.assertEqual(len(parsed_output[k]), 10)

        for k in self.output_key_3d_atoms_lst:
            self.assertEqual(np.array(parsed_output[k]).shape, (10, 32, 3))

        for k in self.output_key_1d_atoms_lst:
            self.assertEqual(np.array(parsed_output[k]).shape, (10, 32))

        for k in self.output_key_9_lst:
            self.assertEqual(np.array(parsed_output[k]).shape, (10, 3, 3))

    def test_calc_md_langevin_nvt(self):
        calc_kwargs = {
            "temperature": 500.0,
            "n_ionic_steps": 1000,
            "n_print": 100,
            "langevin": True,
        }
        shell_output, parsed_output, job_crashed = lammps_file_interface_function(
            working_directory=self.working_directory,
            structure=self.structure,
            potential=self.potential,
            calc_mode="md",
            calc_kwargs=calc_kwargs,
            units=self.units,
        )
        for k in self.output_key_lst:
            self.assertTrue(k in parsed_output.keys())
            self.assertEqual(len(parsed_output[k]), 10)

        for k in self.output_key_3d_atoms_lst:
            self.assertEqual(np.array(parsed_output[k]).shape, (10, 32, 3))

        for k in self.output_key_1d_atoms_lst:
            self.assertEqual(np.array(parsed_output[k]).shape, (10, 32))

        for k in self.output_key_9_lst:
            self.assertEqual(np.array(parsed_output[k]).shape, (10, 3, 3))

    def test_calc_md_langevin_npt(self):
        calc_kwargs = {
            "temperature": 500.0,
            "pressure": 0.0,
            "n_ionic_steps": 1000,
            "n_print": 100,
            "langevin": True,
        }
        shell_output, parsed_output, job_crashed = lammps_file_interface_function(
            working_directory=self.working_directory,
            structure=self.structure,
            potential=self.potential,
            calc_mode="md",
            calc_kwargs=calc_kwargs,
            units=self.units,
        )
        for k in self.output_key_lst:
            self.assertTrue(k in parsed_output.keys())
            self.assertEqual(len(parsed_output[k]), 10)

        for k in self.output_key_3d_atoms_lst:
            self.assertEqual(np.array(parsed_output[k]).shape, (10, 32, 3))

        for k in self.output_key_1d_atoms_lst:
            self.assertEqual(np.array(parsed_output[k]).shape, (10, 32))

        for k in self.output_key_9_lst:
            self.assertEqual(np.array(parsed_output[k]).shape, (10, 3, 3))
