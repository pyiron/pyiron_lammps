from ase.build import bulk
import numpy as np
import os
import unittest
from pyiron_lammps.output import remap_indices, _parse_dump
from pyiron_lammps.structure import UnfoldingPrism


class TestLammpsOutput(unittest.TestCase):
    def setUp(self):
        self.static_folder = os.path.abspath(os.path.join(__file__, "..", "static"))

    def test_remap_indices(self):
        structure = bulk("Ag", cubic=True).repeat([2, 2, 2])
        structure.set_chemical_symbols(
            [
                "Ag",
                "Ag",
                "Ag",
                "Ag",
                "Au",
                "Au",
                "Au",
                "Au",
                "Cu",
                "Cu",
                "Cu",
                "Cu",
                "Ag",
                "Ag",
                "Ag",
                "Ag",
                "Ag",
                "Ag",
                "Ag",
                "Ag",
                "Au",
                "Au",
                "Au",
                "Au",
                "Cu",
                "Cu",
                "Cu",
                "Cu",
                "Ag",
                "Ag",
                "Ag",
                "Ag",
            ]
        )
        ind = remap_indices(
            lammps_indices=[
                1,
                1,
                1,
                1,
                5,
                5,
                5,
                5,
                3,
                3,
                3,
                3,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                5,
                5,
                5,
                5,
                3,
                3,
                3,
                3,
                1,
                1,
                1,
                1,
            ],
            potential_elements=["Ag", "Al", "Cu", "Co", "Au"],
            structure=structure,
        )
        self.assertEqual(sum(ind), 24)
        ind = remap_indices(
            lammps_indices=[2, 2, 2, 2],
            potential_elements=["Ag", "Al", "Cu", "Co", "Au"],
            structure=structure,
        )
        self.assertEqual(sum(ind), 8)
        ind = remap_indices(
            lammps_indices=[2, 2, 2, 2],
            potential_elements=["Au", "Ag", "Cu"],
            structure=structure,
        )
        self.assertEqual(sum(ind), 0)

    def test_dump_chemical(self):
        test_folder = os.path.join(self.static_folder, "dump_chemical")
        structure_ni = bulk("Ni", cubic=True)
        structure_al = bulk("Ni", cubic=True)
        structure_all = bulk("Ni", cubic=True)
        structure_ni.set_chemical_symbols(["H", "Ni", "Ni", "Ni"])
        structure_al.set_chemical_symbols(["H", "Al", "Al", "Al"])
        structure_all.set_chemical_symbols(["Al", "Al", "Ni", "H"])
        for l, s, ind in zip(
            ["dump_NiH.out", "dump_AlH.out", "dump_NiAlH.out"],
            [structure_ni, structure_al, structure_all],
            [np.array([0, 1, 1, 1]), np.array([1, 0, 0, 0]), np.array([0, 0, 1, 2])]
        ):
            output = _parse_dump(
                dump_h5_full_file_name="",
                dump_out_full_file_name=os.path.join(test_folder, l),
                prism=UnfoldingPrism(s.cell),
                structure=s,
                potential_elements=["Ni", "Al", "H"],
            )
            self.assertEqual(output["steps"], [0])
            self.assertEqual(output["natoms"], [4])
            self.assertTrue(np.all(np.equal(output['indices'][0], ind)))
            self.assertTrue(np.all(np.isclose(
                output["forces"],
                [np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])]
            )))
            self.assertTrue(np.all(np.isclose(
                output["velocities"],
                [np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])]
            )))
