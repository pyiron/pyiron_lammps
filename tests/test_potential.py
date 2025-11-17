import os
import unittest

from ase.build import bulk
import pandas
from pyiron_lammps.potential import (
    validate_potential_dataframe,
    get_potential_dataframe,
)


class TestPotential(unittest.TestCase):
    def test_validate_potential_dataframe(self):
        with self.assertRaises(ValueError):
            _ = validate_potential_dataframe(potential_dataframe=pandas.DataFrame({}))
        with self.assertRaises(ValueError):
            _ = validate_potential_dataframe(
                potential_dataframe=pandas.DataFrame({"a": [1, 2]})
            )
        with self.assertRaises(TypeError):
            _ = validate_potential_dataframe(potential_dataframe=0)
        series = validate_potential_dataframe(
            potential_dataframe=pandas.DataFrame({"a": [1]})
        )
        self.assertTrue(isinstance(series, pandas.Series))

    def test_get_potential_dataframe(self):
        df = get_potential_dataframe(
            structure=bulk("Al"),
            resource_path=os.path.abspath(
                os.path.join("..", os.path.dirname(__file__), "static", "potential")
            ),
        )
        self.assertEqual(len(df), 1)
