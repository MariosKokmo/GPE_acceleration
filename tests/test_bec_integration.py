"""End-to-end tests for the simulation loop in both coordinate systems.

The unit tests elsewhere exercise the libraries directly. This module drives
``BaseBEC`` and ``FiniteTempBEC`` the way a real run does — parameter
initialisation, the time-stepping loop, the per-snapshot diagnostics and the
final output writing — with small grids and a handful of steps.

That whole path was previously untested for cylindrical grids, which is why a
snapshot writer calling a method that lived on the wrong class could sit in the
codebase without any test noticing: nothing ever called ``evolve()`` on a
cylindrical BEC.

Everything runs inside a temporary directory because the snapshot writers and
output helpers create files next to the working directory.
"""
import contextlib
import io
import logging
import os
import sys
import tempfile
import types
import unittest
from unittest.mock import MagicMock

import numpy as np
import torch

sys.path.append('.')

from src.library.gpe_library import GPELibrary as gpe
from src.library.gpe_cylindrical_library import GPECylindricalLibrary as cyl
from src.models.base_BEC import BaseBEC
from src.models.finite_temp_BEC import FiniteTempBEC


def _make_app():
    """A stand-in for the application object with the flags the loop reads."""
    app = MagicMock()
    app.device = torch.device('cpu')
    app.logger = logging.getLogger('test_bec_integration')
    app.logger.addHandler(logging.NullHandler())
    app.logger.propagate = False
    # MagicMock attributes are truthy by default; these switch on plotting.
    app.phase_imaging = False
    app.write_velocity = False
    return app


def make_cartesian_system(n1=16, n2=4, n3=16, kmax=8, shots=2, u=5.0, k3=0.0):
    """A System-like object carrying a real Cartesian grid."""
    L = 8.0
    dx = np.array([L / n1, 2.0 / n2, L / n3])
    d_x = float(np.prod(dx))
    x_min = np.array([-L / 2, -1.0, -L / 2])
    dp = [2 * np.pi / L, 2 * np.pi / 2.0, 2 * np.pi / L]
    (x1, x2, x3, p1, p2, p3, p_sq, space_grid, p_grid) = gpe.init_grid(
        x_min, dx, dp, n1, n2, n3, 'cpu')
    gx, gy, gz = space_grid
    potential = 0.5 * (gx ** 2 + gy ** 2 + gz ** 2)

    system = types.SimpleNamespace()
    system._coord = 'cartesian'
    system.simulation_parameters = {
        "Grid_resolution": [n1, n2, n3], "kmax": kmax, "dt": 1e-4,
        "omega_ho": 1.0, "shots": shots, "dtau": 5e-3, "a_ho": 1e-6,
        "u": u, "k3": k3, "dx": dx, "d_x": d_x, "x_min": x_min, "dp": dp,
        "Trapping_frequencies": [10.0, 10.0, 10.0],
    }
    system.uext = types.SimpleNamespace(potential=potential)
    system.p_sq = p_sq
    system.p_grid = p_grid
    system.space_axes = [x1, x2, x3]
    system.momentum_axes = [p1, p2, p3]
    system.space_grid = space_grid
    system.center = (x1[n1 // 2], x2[n2 // 2], x3[n3 // 2])
    system._ground_state = gpe.normalize(
        torch.exp(-(gx ** 2 + gy ** 2 + gz ** 2) / 2).to(torch.cdouble), d_x)
    return system


def make_cylindrical_system(n_r=16, n_phi=4, n_z=12, kmax=8, shots=2, u=5.0):
    """A System-like object for a cylindrical grid.

    ``r`` is deliberately absent so BaseBEC builds the grid and the radial
    operators itself, which is the path a real cylindrical run takes.
    """
    r_max, z_min, z_max = 4.0, -3.0, 3.0
    r, phi, z, kz, m_modes, dr, dphi, dz, (gr, gphi, gz) = cyl.init_grid(
        r_max, z_min, z_max, n_r, n_phi, n_z, 'cpu')

    system = types.SimpleNamespace()
    system._coord = 'cylindrical'
    system.simulation_parameters = {
        "Grid_resolution": [n_r, n_phi, n_z], "kmax": kmax, "dt": 1e-4,
        "omega_ho": 1.0, "shots": shots, "dtau": 5e-3, "a_ho": 1e-6,
        "u": u, "k3": 0.0, "r_max": r_max, "z_min": z_min, "z_max": z_max,
        "Trapping_frequencies": [10.0, 10.0],
    }
    system.uext = types.SimpleNamespace(potential=0.5 * (gr ** 2 + gz ** 2))
    system._ground_state = cyl.normalize(
        torch.exp(-(gr ** 2 + gz ** 2) / 2).to(torch.cdouble), r, dr, dphi, dz)
    return system


class TempWorkdirCase(unittest.TestCase):
    """Run each test in a scratch directory; the writers emit files."""

    def setUp(self):
        self._origin = os.getcwd()
        self._tmp = tempfile.mkdtemp(prefix='bec_integration_')
        os.chdir(self._tmp)

    def tearDown(self):
        os.chdir(self._origin)
        import shutil
        shutil.rmtree(self._tmp, ignore_errors=True)

    def build(self, system, cls=BaseBEC, parameters=None):
        """Construct a BEC and prime it, skipping the ground-state file lookup."""
        bec = cls(parameters=parameters or {}, system=system, app=_make_app(),
                  simulation_name='integration')
        bec._initialize_simulation_parameters()
        bec.psi = system._ground_state.clone()
        return bec

    def run_loop(self, bec):
        with contextlib.redirect_stdout(io.StringIO()):
            bec._main_simulation_loop()
        return bec


class TestCartesianSimulationLoop(TempWorkdirCase):
    def test_loop_runs_and_keeps_the_state_finite(self):
        bec = self.run_loop(self.build(make_cartesian_system()))
        self.assertTrue(bool(torch.all(torch.isfinite(torch.abs(bec.psi)))))

    def test_snapshots_are_recorded(self):
        system = make_cartesian_system(kmax=8, shots=2)
        bec = self.run_loop(self.build(system))
        self.assertEqual(len(bec.rms_measurements), 2)
        self.assertEqual(len(bec.energies), 2)

    def test_snapshot_diagnostics_are_finite_and_sane(self):
        bec = self.run_loop(self.build(make_cartesian_system()))
        for rms in bec.rms_measurements.values():
            self.assertTrue(bool(torch.isfinite(rms)))
            self.assertGreater(float(rms), 0.0)
        for entry in bec.energies:
            for key in ('e_kin', 'e_pot', 'e_int', 'E_total'):
                self.assertTrue(bool(torch.isfinite(torch.as_tensor(entry[key]).real)))

    def test_cross_line_buffer_is_filled_to_its_declared_width(self):
        """The buffer is (shots, n1); the profile written into it must match."""
        system = make_cartesian_system(n1=16, n3=16)
        bec = self.run_loop(self.build(system))
        self.assertEqual(tuple(bec.cross_line.shape), (bec.shots, bec.n1))
        self.assertTrue(bool(torch.all(bec.cross_line[0] >= 0)))
        self.assertGreater(float(bec.cross_line[0].sum()), 0.0)

    def test_non_cubic_grid_does_not_break_the_snapshot_buffer(self):
        """With n1 != n3 an incorrectly oriented profile raises a shape error."""
        system = make_cartesian_system(n1=16, n2=4, n3=8)
        bec = self.run_loop(self.build(system))
        self.assertEqual(tuple(bec.cross_line.shape), (bec.shots, 16))

    def test_number_is_conserved_without_losses(self):
        system = make_cartesian_system(k3=0.0)
        bec = self.run_loop(self.build(system))
        norm = float(bec.d_x * torch.sum(torch.abs(bec.psi) ** 2))
        self.assertAlmostEqual(norm, 1.0, places=9)

    def test_three_body_losses_reduce_the_atom_number(self):
        """The template loop folds the loss rate into utot as an imaginary
        part, so atoms are genuinely removed."""
        system = make_cartesian_system(kmax=40, shots=1, k3=500.0)
        bec = self.run_loop(self.build(system))
        norm = float(bec.d_x * torch.sum(torch.abs(bec.psi) ** 2))
        self.assertLess(norm, 0.999)
        self.assertGreater(norm, 0.0)

    def test_loss_rate_is_read_from_the_configuration(self):
        system = make_cartesian_system(k3=123.0)
        self.assertEqual(self.build(system).k3, 123.0)

    def test_missing_loss_rate_defaults_to_lossless(self):
        """A hand-built System or an older config without the key must still run."""
        system = make_cartesian_system(k3=0.0)
        del system.simulation_parameters["k3"]
        bec = self.run_loop(self.build(system))
        self.assertEqual(bec.k3, 0.0)
        self.assertAlmostEqual(
            float(bec.d_x * torch.sum(torch.abs(bec.psi) ** 2)), 1.0, places=9)

    def test_stronger_loss_removes_more_atoms(self):
        norms = []
        for k3 in (0.0, 200.0, 800.0):
            bec = self.run_loop(self.build(make_cartesian_system(kmax=40, shots=1, k3=k3)))
            norms.append(float(bec.d_x * torch.sum(torch.abs(bec.psi) ** 2)))
        self.assertEqual(norms, sorted(norms, reverse=True))

    def test_loss_removes_density_fastest_at_the_peak(self):
        """The rate goes as |psi|^4, so the densest region decays fastest."""
        bec = self.build(make_cartesian_system(kmax=40, shots=1, k3=500.0))
        before = torch.abs(bec.psi) ** 2
        self.run_loop(bec)
        after = torch.abs(bec.psi) ** 2
        peak = tuple(int(i) for i in np.unravel_index(int(torch.argmax(before)), before.shape))
        edge = (0, 0, 0)
        self.assertLess(float(after[peak] / before[peak]),
                        float(after[edge] / before[edge]))

    def test_snapshot_files_are_written(self):
        self.run_loop(self.build(make_cartesian_system()))
        self.assertTrue(os.path.exists('R-000-cd.dat'))

    def test_final_outputs_are_written(self):
        bec = self.run_loop(self.build(make_cartesian_system()))
        with contextlib.redirect_stdout(io.StringIO()):
            bec._write_simulation_outputs()
        self.assertTrue(os.path.exists('cross_line_density.csv'))
        self.assertTrue(os.path.exists('energies.txt'))

    def test_dark_soliton_imprint_runs_from_the_loop(self):
        system = make_cartesian_system(kmax=8, shots=2)
        bec = self.build(system, parameters={
            "dark_soliton": True, "soliton_positions": [0.0],
            "soliton_widths": [1.0], "soliton_axes": [3],
            "soliton_imprint_time": 0,
        })
        self.assertTrue(bec._soliton_enabled)
        self.run_loop(bec)
        self.assertTrue(bec._soliton_imprinted)
        self.assertTrue(bool(torch.all(torch.isfinite(torch.abs(bec.psi)))))


class TestCylindricalSimulationLoop(TempWorkdirCase):
    """The path that had no coverage at all before.

    ``_write_iteration_data_cylindrical`` reaches for diagnostics through the
    library classes; when those were on an unreachable sibling class, every
    cylindrical run died on its first snapshot with AttributeError.
    """

    def test_loop_runs_and_keeps_the_state_finite(self):
        bec = self.run_loop(self.build(make_cylindrical_system()))
        self.assertTrue(bool(torch.all(torch.isfinite(torch.abs(bec.psi)))))

    def test_grid_and_radial_operators_are_built_from_the_parameters(self):
        bec = self.build(make_cylindrical_system())
        self.assertEqual(bec.r.shape, (bec.n_r,))
        self.assertIn(0, bec.eigvecs_dict)
        self.assertEqual(bec.eigvecs_dict[0].shape, (bec.n_r, bec.n_r))

    def test_snapshot_writer_completes(self):
        """Regression: this raised AttributeError on the first snapshot."""
        system = make_cylindrical_system(kmax=8, shots=2)
        bec = self.run_loop(self.build(system))
        self.assertEqual(len(bec.rms_measurements), 2)
        self.assertEqual(len(bec.energies), 2)

    def test_snapshot_diagnostics_are_finite_and_sane(self):
        bec = self.run_loop(self.build(make_cylindrical_system()))
        for rms in bec.rms_measurements.values():
            self.assertTrue(bool(torch.isfinite(rms)))
            self.assertGreater(float(rms), 0.0)
        for entry in bec.energies:
            self.assertTrue(bool(torch.isfinite(torch.as_tensor(entry['E_total']).real)))

    def test_radial_profile_fills_the_cross_line_buffer(self):
        bec = self.run_loop(self.build(make_cylindrical_system()))
        self.assertEqual(tuple(bec.cross_line.shape), (bec.shots, bec.n_r))
        self.assertGreater(float(bec.cross_line[0].sum()), 0.0)

    def test_number_is_conserved_by_the_cylindrical_step(self):
        bec = self.run_loop(self.build(make_cylindrical_system()))
        norm = float(torch.sum(torch.abs(bec.psi) ** 2 * bec.r.reshape(-1, 1, 1))
                     * (bec.dr * bec.dphi * bec.dz))
        self.assertAlmostEqual(norm, 1.0, places=8)

    def test_snapshot_files_are_written(self):
        self.run_loop(self.build(make_cylindrical_system()))
        self.assertTrue(os.path.exists('R-000-cd.dat'))

    def test_final_outputs_are_written(self):
        bec = self.run_loop(self.build(make_cylindrical_system()))
        with contextlib.redirect_stdout(io.StringIO()):
            bec._write_simulation_outputs()
        self.assertTrue(os.path.exists('cross_line_density.csv'))
        self.assertTrue(os.path.exists('energies.txt'))

    def test_solitons_are_declined_for_cylindrical_grids(self):
        bec = self.build(make_cylindrical_system(), parameters={
            "dark_soliton": True, "soliton_positions": [0.0],
            "soliton_widths": [1.0], "soliton_axes": [3],
        })
        self.assertFalse(bec._soliton_enabled)


class TestFiniteTempSimulationLoop(TempWorkdirCase):
    """SGPE runs in both coordinate systems."""

    def test_cartesian_loop_runs_at_zero_temperature(self):
        system = make_cartesian_system(kmax=8, shots=2)
        bec = self.build(system, cls=FiniteTempBEC,
                         parameters={"temperature": 0.0, "damping_coefficient": 0.05})
        self.run_loop(bec)
        self.assertTrue(bool(torch.all(torch.isfinite(torch.abs(bec.psi)))))

    def test_cartesian_loop_runs_with_thermal_noise(self):
        system = make_cartesian_system(kmax=8, shots=2)
        bec = self.build(system, cls=FiniteTempBEC,
                         parameters={"temperature": 0.5, "damping_coefficient": 0.05})
        self.run_loop(bec)
        self.assertTrue(bool(torch.all(torch.isfinite(torch.abs(bec.psi)))))

    def test_cylindrical_loop_runs_with_thermal_noise(self):
        system = make_cylindrical_system(kmax=8, shots=2)
        bec = self.build(system, cls=FiniteTempBEC,
                         parameters={"temperature": 0.5, "damping_coefficient": 0.05})
        self.run_loop(bec)
        self.assertTrue(bool(torch.all(torch.isfinite(torch.abs(bec.psi)))))

    def test_chemical_potential_is_computed_from_the_initial_state(self):
        system = make_cartesian_system(kmax=8, shots=1)
        bec = self.build(system, cls=FiniteTempBEC,
                         parameters={"temperature": 0.0, "damping_coefficient": 0.05})
        self.assertIsNone(bec.mu)
        self.run_loop(bec)
        self.assertIsInstance(bec.mu, float)
        # Physical scale: a few units of hbar*omega_ho, not 1/d_x.
        self.assertGreater(bec.mu, 0.0)
        self.assertLess(bec.mu, 50.0)

    def test_configured_chemical_potential_is_respected(self):
        system = make_cartesian_system(kmax=4, shots=1)
        bec = self.build(system, cls=FiniteTempBEC, parameters={
            "temperature": 0.0, "damping_coefficient": 0.05,
            "chemical_potential": 2.25})
        self.run_loop(bec)
        self.assertEqual(bec.mu, 2.25)

    def test_atom_number_responds_to_the_reservoir(self):
        """μ above the state's own chemical potential must grow the condensate,
        below it must shrink it. A forced norm would make both identical."""
        def final_norm(offset):
            system = make_cartesian_system(kmax=60, shots=1, u=0.0)
            probe = self.build(system, cls=FiniteTempBEC, parameters={
                "temperature": 0.0, "damping_coefficient": 0.1})
            mu_state = gpe.calculate_chemical_potential(
                probe.psi, probe.uext, probe.u,
                (probe.p1, probe.p2, probe.p3), probe.d_x)
            bec = self.build(make_cartesian_system(kmax=60, shots=1, u=0.0),
                             cls=FiniteTempBEC, parameters={
                                 "temperature": 0.0, "damping_coefficient": 0.1,
                                 "chemical_potential": mu_state + offset})
            self.run_loop(bec)
            return float(bec.d_x * torch.sum(torch.abs(bec.psi) ** 2))

        self.assertGreater(final_norm(+0.5), 1.001)
        self.assertLess(final_norm(-0.5), 0.999)

    def test_three_body_losses_reduce_the_atom_number(self):
        """The SGPE applies the loss as its own operator, not through utot."""
        system = make_cartesian_system(kmax=40, shots=1, k3=500.0)
        bec = self.build(system, cls=FiniteTempBEC, parameters={
            "temperature": 0.0, "damping_coefficient": 0.05,
            "chemical_potential": 1.5})
        self.run_loop(bec)
        norm = float(bec.d_x * torch.sum(torch.abs(bec.psi) ** 2))
        self.assertLess(norm, 0.999)
        self.assertGreater(norm, 0.0)

    def test_losses_also_apply_in_cylindrical_geometry(self):
        system = make_cylindrical_system(kmax=40, shots=1)
        system.simulation_parameters["k3"] = 500.0
        bec = self.build(system, cls=FiniteTempBEC, parameters={
            "temperature": 0.0, "damping_coefficient": 0.05,
            "chemical_potential": 1.5})
        self.run_loop(bec)
        norm = float(torch.sum(torch.abs(bec.psi) ** 2 * bec.r.reshape(-1, 1, 1))
                     * (bec.dr * bec.dphi * bec.dz))
        self.assertLess(norm, 0.999)
        self.assertGreater(norm, 0.0)

    def test_loss_is_disabled_when_k3_is_zero(self):
        system = make_cartesian_system(kmax=40, shots=1, k3=0.0)
        bec = self.build(system, cls=FiniteTempBEC, parameters={
            "temperature": 0.0, "damping_coefficient": 0.0,
            "chemical_potential": 1.5})
        self.run_loop(bec)
        self.assertAlmostEqual(
            float(bec.d_x * torch.sum(torch.abs(bec.psi) ** 2)), 1.0, places=8)

    def test_loss_matches_the_template_loop_at_zero_damping(self):
        """With gamma = 0 the SGPE propagator is unitary, so the atom number
        must decay at the same rate as in the plain split-step loop. This pins
        the separate-operator form to the same rate as folding the loss into
        utot — a loss applied twice, or scaled by gamma, would be off by a
        factor rather than by the splitting error.

        They agree to a fraction of a percent rather than exactly: the template
        loop freezes |psi|^4 at the start of the step, while the Strang-split
        version re-evaluates it after the propagator, which is the more
        accurate of the two.
        """
        template = self.run_loop(self.build(
            make_cartesian_system(kmax=30, shots=0, k3=400.0)))
        sgpe = self.build(make_cartesian_system(kmax=30, shots=0, k3=400.0),
                          cls=FiniteTempBEC, parameters={
                              "temperature": 0.0, "damping_coefficient": 0.0,
                              "chemical_potential": 1.5})
        self.run_loop(sgpe)
        n_template = float(template.d_x * torch.sum(torch.abs(template.psi) ** 2))
        n_sgpe = float(sgpe.d_x * torch.sum(torch.abs(sgpe.psi) ** 2))
        self.assertLess(n_template, 0.999)
        self.assertLess(abs(n_template - n_sgpe) / n_template, 0.01)

    def test_loss_does_not_go_through_the_damped_propagator(self):
        """Folding the loss into utot would multiply it by (i + gamma) and add
        a spurious phase; applied as its own operator the decay is independent
        of gamma for a state held at its own chemical potential."""
        norms = []
        for gamma in (0.0, 0.4):
            bec = self.build(make_cartesian_system(kmax=30, shots=0, k3=400.0),
                             cls=FiniteTempBEC, parameters={
                                 "temperature": 0.0, "damping_coefficient": gamma,
                                 "chemical_potential": 1.5})
            self.run_loop(bec)
            norms.append(float(bec.d_x * torch.sum(torch.abs(bec.psi) ** 2)))
        # the damping itself moves the norm, but the loss factor must not be
        # scaled by gamma: the two runs stay within a few percent
        self.assertLess(abs(norms[0] - norms[1]) / norms[0], 0.05)

    def test_zero_temperature_injects_no_noise(self):
        """At T = 0 the loop must be exactly the damped GPE."""
        system = make_cartesian_system(kmax=6, shots=0)
        params = {"temperature": 0.0, "damping_coefficient": 0.05,
                  "chemical_potential": 1.5}
        first = self.run_loop(self.build(system, cls=FiniteTempBEC, parameters=params))
        second = self.run_loop(self.build(make_cartesian_system(kmax=6, shots=0),
                                          cls=FiniteTempBEC, parameters=params))
        self.assertLess(float((first.psi - second.psi).abs().max()), 1e-14)


class TestLegacyBECLosses(TempWorkdirCase):
    """The default model_type="BEC" runs src.models.BEC.BEC, which is the class
    that folds the three-body loss rate into the total potential."""

    def _build_legacy(self, k3):
        from src.models.BEC import BEC
        system = make_cartesian_system(kmax=40, shots=1, k3=k3)
        system.simulation_parameters["Total_simulation_time"] = 0.001
        bec = BEC(parameters={}, system=system, app=_make_app(),
                  simulation_name='legacy')
        bec._initialize_simulation_parameters()
        bec.psi = system._ground_state.clone()
        return bec

    def test_losses_reduce_the_atom_number(self):
        """Regression: renormalising inside the step silently cancelled this."""
        bec = self._build_legacy(k3=500.0)
        with contextlib.redirect_stdout(io.StringIO()):
            bec._main_simulation_loop()
        norm = float(bec.d_x * torch.sum(torch.abs(bec.psi) ** 2))
        self.assertLess(norm, 0.999)
        self.assertGreater(norm, 0.0)

    def test_number_is_conserved_when_losses_are_disabled(self):
        bec = self._build_legacy(k3=0.0)
        with contextlib.redirect_stdout(io.StringIO()):
            bec._main_simulation_loop()
        norm = float(bec.d_x * torch.sum(torch.abs(bec.psi) ** 2))
        self.assertAlmostEqual(norm, 1.0, places=9)

    def test_loss_removes_density_fastest_where_it_is_highest(self):
        """The rate goes as |psi|^4, so the peak must decay faster than the wings."""
        bec = self._build_legacy(k3=500.0)
        before = torch.abs(bec.psi) ** 2
        with contextlib.redirect_stdout(io.StringIO()):
            bec._main_simulation_loop()
        after = torch.abs(bec.psi) ** 2
        peak = tuple(int(i) for i in np.unravel_index(int(torch.argmax(before)), before.shape))
        peak_ratio = float(after[peak] / before[peak])
        edge_ratio = float(after[0, 0, 0] / before[0, 0, 0])
        self.assertLess(peak_ratio, edge_ratio)


class TestLibrarySelection(TempWorkdirCase):
    """Both coordinate systems must bind every library the loop uses."""

    def test_cartesian_bindings(self):
        bec = self.build(make_cartesian_system())
        self.assertTrue(hasattr(bec._lib, 'split_step_step'))
        self.assertTrue(hasattr(bec._gpe2d_lib, 'rms_radius'))
        self.assertTrue(hasattr(bec._gs_lib, 'find_ground_state'))

    def test_cylindrical_bindings(self):
        bec = self.build(make_cylindrical_system())
        self.assertTrue(hasattr(bec._lib, 'split_step_step'))
        self.assertTrue(hasattr(bec._gpe2d_lib, 'rms_radius'),
                        msg="the cylindrical snapshot writer needs rms_radius")
        self.assertTrue(hasattr(bec._gs_lib, 'find_ground_state'))

    def test_every_library_call_made_by_the_snapshot_writers_resolves(self):
        for build in (make_cartesian_system, make_cylindrical_system):
            bec = self.build(build())
            for name in ('calculate_energy_allocation', 'normalize', 'sgpe_step',
                         'generate_thermal_noise', 'calculate_chemical_potential'):
                self.assertTrue(hasattr(bec._lib, name),
                                msg=f"{bec._coord}: _lib is missing {name}")


if __name__ == '__main__':
    unittest.main()
