import unittest
import os
import shutil
from ase.build import bulk
from pyiron_lammps.compatibility.file import (
    lammps_file_interface_function,
    lammps_file_initialization,
)
from pyiron_lammps.compatibility.calculate import calc_md, calc_minimize


class TestCompatibilityFile(unittest.TestCase):
    def setUp(self):
        self.working_dir = os.path.abspath(os.path.join(__file__, "..", "lmp"))
        self.static_path = os.path.abspath(
            os.path.join("..", os.path.dirname(__file__), "static")
        )
        self.structure = bulk("Al", cubic=True).repeat([2, 2, 2])
        self.potential = "1999--Mishin-Y--Al--LAMMPS--ipr1"
        self.units = "metal"
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

    def test_calc_error(self):
        with self.assertRaises(ValueError):
            lammps_file_interface_function(
                working_directory=self.working_dir,
                structure=self.structure,
                potential=self.potential,
                calc_mode="error",
                units=self.units,
            )

    def test_calc_md_npt(self):
        calc_kwargs = {
            "temperature": 500.0,
            "pressure": 0.0,
            "n_ionic_steps": 1000,
            "n_print": 100,
        }
        shell_output, parsed_output, job_crashed = lammps_file_interface_function(
            working_directory=self.working_dir,
            structure=self.structure,
            potential=self.potential,
            calc_mode="md",
            calc_kwargs=calc_kwargs,
            units=self.units,
            lmp_command="cp "
            + str(os.path.join(self.static_path, "compatibility_output"))
            + "/* .",
            resource_path=os.path.join(self.static_path, "potential"),
        )
        self.assertFalse(job_crashed)
        for key in self.keys:
            self.assertIn(key, parsed_output["generic"])
        with open(self.working_dir + "/lmp.in", "r") as f:
            content = f.readlines()
        content_expected = [
            "units metal\n",
            "dimension 3\n",
            "boundary p p p\n",
            "atom_style atomic\n",
            "read_data lammps.data\n",
            "pair_style eam/alloy\n",
            "variable dumptime equal 100 \n",
            "dump 1 all custom ${dumptime} dump.out id type xsu ysu zsu fx fy fz vx vy vz\n",
            'dump_modify 1 sort id format line "%d %d %20.15g %20.15g %20.15g %20.15g %20.15g %20.15g %20.15g %20.15g %20.15g"\n',
            "fix ensemble all npt temp 500.0 500.0 0.1 iso 0.0 0.0 1.0\n",
            "variable thermotime equal 100 \n",
            "timestep 0.001\n",
            "velocity all create 1000.0 80996 dist gaussian\n",
            "thermo_style custom step temp pe etotal pxx pxy pxz pyy pyz pzz vol\n",
            "thermo_modify format float %20.15g\n",
            "thermo ${thermotime}\n",
            "run 1000 \n",
        ]
        for line in content_expected:
            self.assertIn(line, content)

    def test_calc_md_temp(self):
        output = calc_md(temperature=300, units="metal")
        self.assertIn(
            "fix ensemble all nvt temp 300.0 300.0 0.1",
            output[0],
        )

    def test_calc_md_temp_list(self):
        output = calc_md(temperature=[300, 400], units="metal")
        self.assertIn(
            "fix ensemble all nvt temp 300.0 400.0 0.1",
            output[0],
        )

    def test_calc_md_pressure(self):
        output = calc_md(temperature=300, pressure=1.0, units="metal")
        self.assertIn("fix ensemble all npt temp", output[0])
        self.assertIn("iso 10000.0 10000.0 1.0", output[0])

    def test_calc_md_langevin(self):
        output = calc_md(temperature=300, langevin=True, units="metal")
        self.assertIn("fix langevin all langevin", output[1])

    def test_calc_md_delta(self):
        with self.assertWarns(Warning):
            calc_md(temperature=300, delta_temp=100.0, units="metal")
        with self.assertWarns(Warning):
            calc_md(temperature=300, pressure=1.0, delta_press=1000.0, units="metal")

    def test_calc_minimize_new(self):
        output, _ = calc_minimize(structure=bulk("Al"), units="metal")
        self.assertIn("min_style cg", output)

    def test_calc_minimize_pressure_new(self):
        output, _ = calc_minimize(structure=bulk("Al"), pressure=1.0, units="metal")
        self.assertIn("fix ensemble all box/relax iso 10000.0", output[4])

    def test_calc_md_n_ionic_steps(self):
        calc_kwargs = {"temperature": 500.0, "n_ionic_steps": 100, "n_print": 10}
        shell_output, parsed_output, job_crashed = lammps_file_interface_function(
            working_directory=self.working_dir,
            structure=self.structure,
            potential=self.potential,
            calc_mode="md",
            calc_kwargs=calc_kwargs,
            units=self.units,
            lmp_command="cp "
            + str(os.path.join(self.static_path, "compatibility_output"))
            + "/* .",
            resource_path=os.path.join(self.static_path, "potential"),
        )
        self.assertFalse(job_crashed)
        for key in self.keys:
            self.assertIn(key, parsed_output["generic"])
        with open(self.working_dir + "/lmp.in", "r") as f:
            content = f.readlines()
        self.assertIn("run 100 \n", content)

    def test_calc_md_wo_n_ionic_steps(self):
        calc_kwargs = {"temperature": 500.0, "n_print": 10}
        shell_output, parsed_output, job_crashed = lammps_file_interface_function(
            working_directory=self.working_dir,
            structure=self.structure,
            potential=self.potential,
            calc_mode="md",
            calc_kwargs=calc_kwargs,
            units=self.units,
            lmp_command="cp "
            + str(os.path.join(self.static_path, "compatibility_output"))
            + "/* .",
            resource_path=os.path.join(self.static_path, "potential"),
        )
        self.assertFalse(job_crashed)
        for key in self.keys:
            self.assertIn(key, parsed_output["generic"])
        with open(self.working_dir + "/lmp.in", "r") as f:
            content = f.readlines()
        self.assertIn("run 1 \n", content)

    def test_lammps_file_initialization(self):
        init_commands = lammps_file_initialization(structure=self.structure)
        self.assertEqual(init_commands[0], "units metal")
        self.assertEqual(init_commands[1], "dimension 3")
        self.assertEqual(init_commands[2], "boundary p p p")
        self.assertEqual(init_commands[3], "atom_style atomic")
        self.assertEqual(init_commands[4], "read_data lammps.data")
        structure = self.structure.copy()
        structure.pbc = [True, True, False]
        init_commands = lammps_file_initialization(structure=structure)
        self.assertEqual(init_commands[2], "boundary p p f")
        structure.pbc = [True, False, False]
        init_commands = lammps_file_initialization(structure=structure)
        self.assertEqual(init_commands[2], "boundary p f f")
        structure.pbc = [False, False, False]
        init_commands = lammps_file_initialization(structure=structure)
        self.assertEqual(init_commands[2], "boundary f f f")

    def test_calc_md_nvt(self):
        calc_kwargs = {"temperature": 500.0, "n_print": 100}
        shell_output, parsed_output, job_crashed = lammps_file_interface_function(
            working_directory=self.working_dir,
            structure=self.structure,
            potential=self.potential,
            calc_mode="md",
            calc_kwargs=calc_kwargs,
            units=self.units,
            lmp_command="cp "
            + str(os.path.join(self.static_path, "compatibility_output"))
            + "/* .",
            resource_path=os.path.join(self.static_path, "potential"),
        )
        self.assertFalse(job_crashed)
        for key in self.keys:
            self.assertIn(key, parsed_output["generic"])
        with open(self.working_dir + "/lmp.in", "r") as f:
            content = f.readlines()
        content_expected = [
            "units metal\n",
            "dimension 3\n",
            "boundary p p p\n",
            "atom_style atomic\n",
            "read_data lammps.data\n",
            "pair_style eam/alloy\n",
            "variable dumptime equal 100 \n",
            "dump 1 all custom ${dumptime} dump.out id type xsu ysu zsu fx fy fz vx vy vz\n",
            'dump_modify 1 sort id format line "%d %d %20.15g %20.15g %20.15g %20.15g %20.15g %20.15g %20.15g %20.15g %20.15g"\n',
            "fix ensemble all nvt temp 500.0 500.0 0.1\n",
            "variable thermotime equal 100 \n",
            "timestep 0.001\n",
            "velocity all create 1000.0 80996 dist gaussian\n",
            "thermo_style custom step temp pe etotal pxx pxy pxz pyy pyz pzz vol\n",
            "thermo_modify format float %20.15g\n",
            "thermo ${thermotime}\n",
            "run 1 \n",
        ]
        for line in content_expected:
            self.assertIn(line, content)

    def test_calc_static(self):
        shell_output, parsed_output, job_crashed = lammps_file_interface_function(
            working_directory=self.working_dir,
            structure=self.structure,
            potential=self.potential,
            calc_mode="static",
            units=self.units,
            lmp_command="cp "
            + str(os.path.join(self.static_path, "compatibility_output"))
            + "/* .",
            resource_path=os.path.join(self.static_path, "potential"),
        )
        self.assertFalse(job_crashed)
        for key in self.keys:
            self.assertIn(key, parsed_output["generic"])
        with open(self.working_dir + "/lmp.in", "r") as f:
            content = f.readlines()
        content_expected = [
            "units metal\n",
            "dimension 3\n",
            "boundary p p p\n",
            "atom_style atomic\n",
            "read_data lammps.data\n",
            "pair_style eam/alloy\n",
            "variable dumptime equal 1 \n",
            "dump 1 all custom ${dumptime} dump.out id type xsu ysu zsu fx fy fz vx vy vz\n",
            'dump_modify 1 sort id format line "%d %d %20.15g %20.15g %20.15g %20.15g %20.15g %20.15g %20.15g %20.15g %20.15g"\n',
            "variable thermotime equal 1\n",
            "thermo_style custom step temp pe etotal pxx pxy pxz pyy pyz pzz vol\n",
            "thermo_modify format float %20.15g\n",
            "thermo ${thermotime}\n",
            "run 0\n",
        ]
        for line in content_expected:
            self.assertIn(line, content)

    def test_calc_minimize(self):
        shell_output, parsed_output, job_crashed = lammps_file_interface_function(
            working_directory=self.working_dir,
            structure=self.structure,
            potential=self.potential,
            calc_mode="minimize",
            units=self.units,
            lmp_command="cp "
            + str(os.path.join(self.static_path, "compatibility_output"))
            + "/* .",
            resource_path=os.path.join(self.static_path, "potential"),
        )
        self.assertFalse(job_crashed)
        for key in self.keys:
            self.assertIn(key, parsed_output["generic"])
        with open(self.working_dir + "/lmp.in", "r") as f:
            content = f.readlines()
        content_expected = [
            "units metal\n",
            "dimension 3\n",
            "boundary p p p\n",
            "atom_style atomic\n",
            "read_data lammps.data\n",
            "pair_style eam/alloy\n",
            "variable dumptime equal 1 \n",
            "dump 1 all custom ${dumptime} dump.out id type xsu ysu zsu fx fy fz vx vy vz\n",
            'dump_modify 1 sort id format line "%d %d %20.15g %20.15g %20.15g %20.15g %20.15g %20.15g %20.15g %20.15g %20.15g"\n',
            "variable thermotime equal 100 \n",
            "thermo_style custom step temp pe etotal pxx pxy pxz pyy pyz pzz vol\n",
            "thermo_modify format float %20.15g\n",
            "thermo ${thermotime}\n",
            "min_style cg\n",
            "minimize 0.0 0.0001 100000 10000000\n",
        ]
        for line in content_expected:
            self.assertIn(line, content)

    def test_calc_minimize_pressure(self):
        shell_output, parsed_output, job_crashed = lammps_file_interface_function(
            working_directory=self.working_dir,
            structure=self.structure,
            potential=self.potential,
            calc_mode="minimize",
            units=self.units,
            calc_kwargs={"pressure": 0.0},
            lmp_command="cp "
            + str(os.path.join(self.static_path, "compatibility_output"))
            + "/* .",
            resource_path=os.path.join(self.static_path, "potential"),
        )
        self.assertFalse(job_crashed)
        for key in self.keys:
            self.assertIn(key, parsed_output["generic"])
        with open(self.working_dir + "/lmp.in", "r") as f:
            content = f.readlines()
        content_expected = [
            "units metal\n",
            "dimension 3\n",
            "boundary p p p\n",
            "atom_style atomic\n",
            "read_data lammps.data\n",
            "pair_style eam/alloy\n",
            "variable dumptime equal 1 \n",
            "dump 1 all custom ${dumptime} dump.out id type xsu ysu zsu fx fy fz vx vy vz\n",
            'dump_modify 1 sort id format line "%d %d %20.15g %20.15g %20.15g %20.15g %20.15g %20.15g %20.15g %20.15g %20.15g"\n',
            "variable thermotime equal 100 \n",
            "thermo_style custom step temp pe etotal pxx pxy pxz pyy pyz pzz vol\n",
            "thermo_modify format float %20.15g\n",
            "thermo ${thermotime}\n",
            "fix ensemble all box/relax iso 0.0\n",
            "min_style cg\n",
            "minimize 0.0 0.0001 100000 10000000\n",
        ]
        for line in content_expected:
            self.assertIn(line, content)
