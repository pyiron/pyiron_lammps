import unittest
import numpy as np
from ase.build import bulk
from pyiron_lammps.structure import structure_to_lammps


class TestLammpsStructure(unittest.TestCase):
    def test_structure_to_lammps_with_velocity(self):
        structure = bulk("Al")
        structure.set_velocities([[1.0, 1.0, 1.0]])
        structure_lammps = structure_to_lammps(structure=structure)
        self.assertEqual(len(structure), len(structure_lammps))
        self.assertTrue(np.all(np.isclose(
            structure_lammps.cell,
            np.array([
                [2.8637824638055176, 0.0, 0.0],
                [-1.4318912319, 2.4801083645, 0.0],
                [1.4318912319, 0.8267027881, 2.3382685902],
            ])
        )))
        self.assertTrue(np.all(np.isclose(
            structure_lammps.get_velocities(),
            np.array([[1.41421356, 0.81649658, 0.57735027]]),
        )))

    def test_structure_to_lammps_without_velocity(self):
        structure = bulk("Al")
        structure_lammps = structure_to_lammps(structure=structure)
        self.assertEqual(len(structure), len(structure_lammps))
        self.assertTrue(np.all(np.isclose(
            structure_lammps.cell,
            np.array([
                [2.8637824638055176, 0.0, 0.0],
                [-1.4318912319, 2.4801083645, 0.0],
                [1.4318912319, 0.8267027881, 2.3382685902],
            ])
        )))
