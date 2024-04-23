import unittest

from ase.build import bulk
import pandas

from pyiron_lammps.parallel import combine_structure_and_potential


class TestErrorMessages(unittest.TestCase):
    def test_combine_structure_and_potential(self):
        structure = bulk("Al", cubic=True)
        potential_dataframe = pandas.DataFrame({"a": [1], "b": [2]})
        with self.assertRaises(TypeError):
            combine_structure_and_potential(
                structure=1,
                potential_dataframe=potential_dataframe
            )
        with self.assertRaises(ValueError):
            combine_structure_and_potential(
                structure=[structure],
                potential_dataframe=[potential_dataframe, potential_dataframe]
            )
        with self.assertRaises(TypeError):
            combine_structure_and_potential(
                structure=[structure],
                potential_dataframe=1
            )
        with self.assertRaises(TypeError):
            combine_structure_and_potential(
                structure=structure,
                potential_dataframe=1
            )
