from ase.build import bulk
import numpy as np
import os
import unittest
from pyiron_lammps.output import remap_indices_ase, parse_lammps_output, _parse_dump
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
        ind = remap_indices_ase(
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
        ind = remap_indices_ase(
            lammps_indices=[2, 2, 2, 2],
            potential_elements=["Ag", "Al", "Cu", "Co", "Au"],
            structure=structure,
        )
        self.assertEqual(sum(ind), 8)
        ind = remap_indices_ase(
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
            [np.array([0, 1, 1, 1]), np.array([1, 0, 0, 0]), np.array([0, 0, 1, 2])],
        ):
            output = _parse_dump(
                dump_h5_full_file_name="",
                dump_out_full_file_name=os.path.join(test_folder, l),
                prism=UnfoldingPrism(s.cell),
                structure=s,
                potential_elements=["Ni", "Al", "H"],
                remap_indices_funct=remap_indices_ase,
            )
            self.assertEqual(output["steps"], [0])
            self.assertEqual(output["natoms"], [4])
            self.assertTrue(np.all(np.equal(output["indices"][0], ind)))
            self.assertTrue(
                np.all(
                    np.isclose(
                        output["forces"],
                        [
                            np.array(
                                [
                                    [0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0],
                                ]
                            )
                        ],
                    )
                )
            )
            self.assertTrue(
                np.all(
                    np.isclose(
                        output["velocities"],
                        [
                            np.array(
                                [
                                    [0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0],
                                ]
                            )
                        ],
                    )
                )
            )

    def test_full_job_output(self):
        test_folder = os.path.join(self.static_folder, "full_job")
        structure_ni = bulk("Ni", cubic=True)
        output_dict = parse_lammps_output(
            working_directory=test_folder,
            structure=structure_ni,
            potential_elements=["Ni", "Al", "H"],
            units="metal",
            prism=UnfoldingPrism(structure_ni.cell),
            dump_h5_file_name="dump.h5",
            dump_out_file_name="dump.out",
            log_lammps_file_name="log.lammps",
            remap_indices_funct=remap_indices_ase,
        )
        self.assertEqual(output_dict["generic"]["steps"], np.array([0]))
        self.assertEqual(output_dict["generic"]["natoms"], np.array([4.0]))
        self.assertTrue(
            np.all(
                np.isclose(
                    output_dict["generic"]["energy_pot"], np.array([-17.80000005])
                )
            )
        )
        self.assertTrue(
            np.all(
                np.isclose(
                    output_dict["generic"]["energy_tot"], np.array([-17.80000005])
                )
            )
        )
        self.assertTrue(
            np.all(np.isclose(output_dict["generic"]["volume"], np.array([43.614208])))
        )
        self.assertEqual(output_dict["generic"]["temperature"], np.array([0.0]))
        self.assertTrue(
            np.all(
                np.isclose(
                    output_dict["generic"]["pressures"],
                    np.array(
                        [
                            [
                                [5.38768850e-05, -4.07839310e-16, -1.83528844e-15],
                                [-4.07839310e-16, 5.38768850e-05, -1.01960322e-15],
                                [-1.83528844e-15, -1.01960322e-15, 5.38768850e-05],
                            ]
                        ]
                    ),
                )
            )
        )
        self.assertTrue(
            np.all(
                np.isclose(
                    output_dict["generic"]["positions"],
                    np.array(
                        [
                            [
                                [0.00000000e00, 0.00000000e00, 0.00000000e00],
                                [0.00000000e00, 1.76000000e00, 1.76000000e00],
                                [1.76000000e00, 1.07768918e-16, 1.76000000e00],
                                [1.76000000e00, 1.76000000e00, 2.15537837e-16],
                            ]
                        ]
                    ),
                )
            )
        )
        self.assertTrue(
            np.all(
                np.isclose(
                    output_dict["generic"]["unwrapped_positions"],
                    np.array(
                        [
                            [
                                [0.00000000e00, 0.00000000e00, 0.00000000e00],
                                [0.00000000e00, 1.76000000e00, 1.76000000e00],
                                [1.76000000e00, 1.07768918e-16, 1.76000000e00],
                                [1.76000000e00, 1.76000000e00, 2.15537837e-16],
                            ]
                        ]
                    ),
                )
            )
        )
        self.assertTrue(
            np.all(
                np.isclose(
                    output_dict["generic"]["velocities"],
                    np.array(
                        [
                            [
                                [0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0],
                            ]
                        ]
                    ),
                )
            )
        )
        self.assertTrue(
            np.all(
                np.isclose(
                    output_dict["generic"]["forces"],
                    np.array(
                        [
                            [
                                [-2.22044605e-16, -1.38777878e-17, -5.55111512e-17],
                                [-5.55111512e-17, -1.66533454e-16, 6.93889390e-17],
                                [-5.55111512e-17, -3.39907768e-33, -2.08166817e-16],
                                [0.00000000e00, -6.93889390e-18, -2.42861287e-16],
                            ]
                        ]
                    ),
                )
            )
        )
        self.assertTrue(
            np.all(
                np.isclose(
                    output_dict["generic"]["cells"],
                    np.array(
                        [
                            [
                                [3.52000000e00, 2.15537837e-16, 2.15537837e-16],
                                [0.00000000e00, 3.52000000e00, 2.15537837e-16],
                                [0.00000000e00, 0.00000000e00, 3.52000000e00],
                            ]
                        ]
                    ),
                )
            )
        )
        self.assertTrue(
            np.all(
                np.isclose(
                    output_dict["generic"]["indices"],
                    np.array([[0, 0, 0, 0]]),
                )
            )
        )
