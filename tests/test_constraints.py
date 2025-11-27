import unittest

from ase.build import bulk
from ase.constraints import FixAtoms, FixCom, FixedPlane

from pyiron_lammps.compatibility.constraints import (
    set_selective_dynamics,
)


class TestConstraints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        structure = bulk("Cu", cubic=True)
        structure.symbols[2:] = "Al"
        cls.structure = structure

    def test_selective_dynamics_mixed_calcmd_y(self):
        atoms = self.structure.copy()
        c1 = FixAtoms(indices=[atom.index for atom in atoms if atom.symbol == "Cu"])
        c2 = FixedPlane(
            [atom.index for atom in atoms if atom.symbol == "Al"],
            [0, 1, 0],
        )
        atoms.set_constraint([c1, c2])
        control_dict = set_selective_dynamics(structure=atoms, calc_md=True)
        self.assertEqual(len(control_dict), 6)
        self.assertTrue(control_dict["group constraintxyz"], "id 1 2")
        self.assertTrue(
            control_dict["fix constraintxyz"], "constraintxyz setforce 0.0 0.0 0.0"
        )
        self.assertTrue(control_dict["velocity constraintxyz"], "set 0.0 0.0 0.0")
        self.assertTrue(control_dict["group constrainty"], "id 3 4")
        self.assertTrue(
            control_dict["fix constrainty"], "constrainty setforce NULL 0.0 NULL"
        )
        self.assertTrue(control_dict["velocity constrainty"], "set NULL 0.0 NULL")

    def test_selective_dynamics_mixed_x(self):
        atoms = self.structure.copy()
        c1 = FixAtoms(indices=[atom.index for atom in atoms if atom.symbol == "Cu"])
        c2 = FixedPlane(
            [atom.index for atom in atoms if atom.symbol == "Al"],
            [1, 0, 0],
        )
        atoms.set_constraint([c1, c2])
        control_dict = set_selective_dynamics(structure=atoms, calc_md=False)
        self.assertEqual(len(control_dict), 4)
        self.assertTrue(control_dict["group constraintxyz"], "id 1 2")
        self.assertTrue(
            control_dict["fix constraintxyz"], "constraintxyz setforce 0.0 0.0 0.0"
        )
        self.assertTrue(control_dict["group constraintx"], "id 3 4")
        self.assertTrue(
            control_dict["fix constraintx"], "constraintx setforce 0.0 NULL NULL"
        )

    def test_selective_dynamics_single_fix(self):
        atoms = self.structure.copy()
        c1 = FixAtoms(indices=[atom.index for atom in atoms if atom.symbol == "Cu"])
        atoms.set_constraint(c1)
        control_dict = set_selective_dynamics(structure=atoms, calc_md=False)
        self.assertEqual(len(control_dict), 2)
        self.assertTrue(control_dict["group constraintxyz"], "id 1 2")
        self.assertTrue(
            control_dict["fix constraintxyz"], "constraintxyz setforce 0.0 0.0 0.0"
        )

    def test_selective_dynamics_errors(self):
        atoms = self.structure.copy()
        atoms.set_constraint(FixCom())
        with self.assertRaises(ValueError):
            set_selective_dynamics(structure=atoms, calc_md=False)

    def test_selective_dynamics_wrong_plane(self):
        atoms = self.structure.copy()
        atoms.set_constraint(
            FixedPlane(
                [atom.index for atom in atoms if atom.symbol == "Al"],
                [2, 1, 0],
            )
        )
        with self.assertRaises(ValueError):
            set_selective_dynamics(structure=atoms, calc_md=False)
