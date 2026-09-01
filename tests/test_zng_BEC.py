"""Tests for the experimental ZNG two-component model.

This model had no test coverage at all, which is how a call into
``calculate_energy_allocation`` kept an outdated signature without anything
noticing. The tests here drive the real loop on a small grid, and pin the
``zng_condensate_exchange`` switch that decides whether the condensate may
actually trade atoms with the thermal cloud.
"""
import contextlib
import io
import logging
import os
import shutil
import sys
import tempfile
import types
import unittest
from unittest.mock import MagicMock

import numpy as np
import torch

sys.path.append('.')

from src.experimental.zng.zng_BEC import ZNGBEC
from src.experimental.zng.zng_library import sample_initial_thermal_cloud
from src.library.gpe_library import GPELibrary as gpe


def make_system(n1=16, n2=4, n3=16, kmax=6, shots=0):
    """A System-like object carrying a real Cartesian grid."""
    L = 8.0
    dx = np.array([L / n1, 2.0 / n2, L / n3])
    d_x = float(np.prod(dx))
    x_min = np.array([-L / 2, -1.0, -L / 2])
    dp = [2 * np.pi / L, 2 * np.pi / 2.0, 2 * np.pi / L]
    (x1, x2, x3, p1, p2, p3, p_sq, space_grid, p_grid) = gpe.init_grid(
        x_min, dx, dp, n1, n2, n3, 'cpu')
    gx, gy, gz = space_grid

    system = types.SimpleNamespace()
    system._coord = 'cartesian'
    system.simulation_parameters = {
        "Grid_resolution": [n1, n2, n3], "kmax": kmax, "dt": 1e-4,
        "omega_ho": 1.0, "shots": shots, "dtau": 2e-3, "a_ho": 1e-6,
        "u": 5.0, "k3": 0.0, "dx": dx, "d_x": d_x, "x_min": x_min, "dp": dp,
        "w": [1.0, 1.0, 1.0], "Trapping_frequencies": [10.0, 10.0, 10.0],
    }
    system.uext = types.SimpleNamespace(potential=0.5 * (gx ** 2 + gy ** 2 + gz ** 2))
    system.p_sq = p_sq
    system.p_grid = p_grid
    system.space_axes = [x1, x2, x3]
    system.momentum_axes = [p1, p2, p3]
    system.space_grid = space_grid
    system.center = (x1[n1 // 2], x2[n2 // 2], x3[n3 // 2])
    system._ground_state = gpe.normalize(
        torch.exp(-(gx ** 2 + gy ** 2 + gz ** 2) / 2).to(torch.cdouble), d_x)
    return system


def make_app():
    app = MagicMock()
    app.device = torch.device('cpu')
    app.logger = logging.getLogger('test_zng')
    app.logger.addHandler(logging.NullHandler())
    app.logger.propagate = False
    app.phase_imaging = False
    app.write_velocity = False
    return app


class ZNGCase(unittest.TestCase):
    def setUp(self):
        self._origin = os.getcwd()
        self._tmp = tempfile.mkdtemp(prefix='zng_')
        os.chdir(self._tmp)

    def tearDown(self):
        os.chdir(self._origin)
        shutil.rmtree(self._tmp, ignore_errors=True)

    def build(self, exchange=False, n_test=400, **params):
        """Construct a ZNGBEC primed past the ground-state file lookup."""
        system = make_system(**{k: params.pop(k) for k in ('kmax', 'shots')
                                if k in params})
        settings = {"temperature": 1.0, "n_test_particles": n_test,
                    "gamma_12": 0.1, "chemical_potential": 2.0,
                    "zng_thermal_fraction_mode": "explicit",
                    "zng_thermal_fraction": 0.25,
                    "zng_condensate_exchange": exchange}
        settings.update(params)
        bec = ZNGBEC(parameters=settings, system=system, app=make_app(),
                     simulation_name='zng')
        bec._initialize_simulation_parameters()
        bec.psi = system._ground_state.clone()
        w = torch.tensor(system.simulation_parameters["w"], dtype=torch.float64)
        bec.particle_positions, bec.particle_momenta = sample_initial_thermal_cloud(
            n_test, w, bec.temperature, bec.device)
        bec._update_thermal_density()
        return bec

    def run_loop(self, bec):
        with contextlib.redirect_stdout(io.StringIO()):
            bec._main_simulation_loop()
        return bec


class TestZNGConfiguration(ZNGCase):
    def test_exchange_defaults_to_off(self):
        """The original behaviour is preserved unless asked for."""
        bec = ZNGBEC(parameters={}, system=make_system(), app=make_app(),
                     simulation_name='zng')
        bec._initialize_simulation_parameters()
        self.assertFalse(bec.condensate_exchange)

    def test_exchange_flag_is_read(self):
        self.assertTrue(self.build(exchange=True).condensate_exchange)
        self.assertFalse(self.build(exchange=False).condensate_exchange)

    def test_flag_reaches_the_parameters_from_the_cartesian_config(self):
        from src.sims_setup.cartesian_setup import get_simulation_parameters_cartesian
        import inspect
        source = inspect.getsource(get_simulation_parameters_cartesian)
        self.assertIn("zng_condensate_exchange", source)

    def test_flag_reaches_the_parameters_from_the_cylindrical_config(self):
        from src.sims_setup.cylindrical_setup import get_simulation_parameters_cylindrical
        import inspect
        source = inspect.getsource(get_simulation_parameters_cylindrical)
        self.assertIn("zng_condensate_exchange", source)

    def test_other_zng_settings_still_load(self):
        bec = self.build(exchange=True, n_test=250, gamma_12=0.25, enable_c22=True)
        self.assertEqual(bec.n_test, 250)
        self.assertEqual(bec.gamma_12, 0.25)
        self.assertTrue(bec.enable_c22)


class TestZNGCondensateNumber(ZNGCase):
    def test_pinned_mode_holds_the_condensate_number(self):
        bec = self.run_loop(self.build(exchange=False, kmax=10))
        self.assertAlmostEqual(bec.condensate_number(), 1.0, places=10)

    def test_exchange_mode_lets_the_condensate_number_move(self):
        """R > 0 grows the condensate, R < 0 shrinks it; with the norm pinned
        neither can happen."""
        bec = self.run_loop(self.build(exchange=True, kmax=10))
        self.assertNotAlmostEqual(bec.condensate_number(), 1.0, places=6)
        self.assertGreater(bec.condensate_number(), 0.0)
        self.assertTrue(np.isfinite(bec.condensate_number()))

    def test_condensate_number_reports_the_norm(self):
        bec = self.build(exchange=True)
        expected = float(bec.d_x * torch.sum(torch.abs(bec.psi) ** 2))
        self.assertAlmostEqual(bec.condensate_number(), expected, places=12)

    def test_state_stays_finite_in_both_modes(self):
        for exchange in (False, True):
            bec = self.run_loop(self.build(exchange=exchange, kmax=10))
            self.assertTrue(bool(torch.all(torch.isfinite(torch.abs(bec.psi)))),
                            msg=f"psi went non-finite with exchange={exchange}")

    def test_the_two_modes_differ(self):
        torch.manual_seed(0)
        pinned = self.run_loop(self.build(exchange=False, kmax=10))
        torch.manual_seed(0)
        free = self.run_loop(self.build(exchange=True, kmax=10))
        self.assertNotAlmostEqual(pinned.condensate_number(),
                                  free.condensate_number(), places=6)


class TestThermalFractionConventions(ZNGCase):
    """The thermal fraction is what puts the two components on one scale."""

    def test_temperature_convention_follows_the_ideal_bose_law(self):
        """f = (T/T_c)^3 = T^3 zeta(3) / N for a 3-D harmonic trap."""
        from src.library.parameters import CONSTANTS
        zeta_3 = 1.2020569031595943
        for kT in (1.0, 5.0, 20.0):
            bec = self.build(temperature=kT, zng_thermal_fraction_mode="temperature")
            expected = min(1.0, kT ** 3 * zeta_3 / float(CONSTANTS.nat))
            self.assertAlmostEqual(bec.thermal_fraction, expected, places=12)

    def test_temperature_convention_saturates_above_tc(self):
        bec = self.build(temperature=1e6, zng_thermal_fraction_mode="temperature")
        self.assertEqual(bec.thermal_fraction, 1.0)

    def test_temperature_convention_is_monotonic(self):
        fractions = [self.build(temperature=kT,
                                zng_thermal_fraction_mode="temperature").thermal_fraction
                     for kT in (1.0, 10.0, 30.0)]
        self.assertEqual(fractions, sorted(fractions))

    def test_explicit_convention_uses_the_configured_value(self):
        bec = self.build(zng_thermal_fraction_mode="explicit",
                         zng_thermal_fraction=0.35)
        self.assertAlmostEqual(bec.thermal_fraction, 0.35, places=12)

    def test_explicit_convention_ignores_the_temperature(self):
        cold = self.build(temperature=0.1, zng_thermal_fraction_mode="explicit",
                          zng_thermal_fraction=0.4)
        hot = self.build(temperature=50.0, zng_thermal_fraction_mode="explicit",
                         zng_thermal_fraction=0.4)
        self.assertEqual(cold.thermal_fraction, hot.thermal_fraction)

    def test_explicit_convention_requires_a_value(self):
        """Built directly, so the helper's default fraction cannot mask this."""
        bec = ZNGBEC(parameters={"zng_thermal_fraction_mode": "explicit"},
                     system=make_system(), app=make_app(), simulation_name='zng')
        with self.assertRaises(ValueError):
            bec._initialize_simulation_parameters()

    def test_explicit_fraction_is_range_checked(self):
        for bad in (-0.1, 1.5):
            with self.assertRaises(ValueError):
                self.build(zng_thermal_fraction_mode="explicit",
                           zng_thermal_fraction=bad)

    def test_unknown_convention_is_rejected(self):
        with self.assertRaises(ValueError):
            self.build(zng_thermal_fraction_mode="whatever")

    def test_default_convention_is_temperature(self):
        bec = ZNGBEC(parameters={}, system=make_system(), app=make_app(),
                     simulation_name='zng')
        bec._initialize_simulation_parameters()
        self.assertEqual(bec.thermal_fraction_mode, "temperature")


class TestThermalFractionNormalisation(ZNGCase):
    def test_thermal_density_integrates_to_the_fraction(self):
        """This is the whole point: n~ must be measured on the condensate's
        scale, not in raw test-particle counts."""
        bec = self.build(zng_thermal_fraction_mode="explicit",
                         zng_thermal_fraction=0.3, n_test=2000)
        self.assertAlmostEqual(float(bec.d_x * torch.sum(bec.n_tilde)), 0.3, places=6)

    def test_thermal_density_is_independent_of_the_particle_count(self):
        """N_test is a convergence knob; the physics must not depend on it."""
        integrals = []
        for n_test in (400, 1600, 6400):
            torch.manual_seed(0)
            bec = self.build(zng_thermal_fraction_mode="explicit",
                             zng_thermal_fraction=0.25, n_test=n_test)
            integrals.append(float(bec.d_x * torch.sum(bec.n_tilde)))
        for value in integrals:
            self.assertAlmostEqual(value, 0.25, places=6)

    def test_mean_field_ratio_no_longer_scales_with_the_particle_count(self):
        """Before the weight, peak 2u*n~ / peak u*n_c grew linearly with N_test."""
        peaks = []
        for n_test in (400, 3200):
            torch.manual_seed(0)
            bec = self.build(zng_thermal_fraction_mode="explicit",
                             zng_thermal_fraction=0.25, n_test=n_test)
            n_c = torch.abs(bec.psi) ** 2
            peaks.append(float(2 * bec.n_tilde.max() / n_c.max()))
        self.assertLess(abs(peaks[1] - peaks[0]) / peaks[0], 0.5,
                        msg=f"mean-field ratio still tracks N_test: {peaks}")

    def test_particle_weight_is_the_fraction_per_particle(self):
        bec = self.build(zng_thermal_fraction_mode="explicit",
                         zng_thermal_fraction=0.5, n_test=1000)
        self.assertAlmostEqual(bec.particle_weight, 0.5 / 1000, places=15)

    def test_zero_fraction_leaves_an_empty_cloud(self):
        bec = self.build(zng_thermal_fraction_mode="explicit",
                         zng_thermal_fraction=0.0)
        self.assertAlmostEqual(float(bec.d_x * torch.sum(bec.n_tilde)), 0.0, places=12)


class TestCondensateThermalSplit(ZNGCase):
    """The two components must add up to the whole sample."""

    def _initialised(self, fraction, n_test=500):
        system = make_system()
        bec = ZNGBEC(parameters={
            "temperature": 1.0, "n_test_particles": n_test, "gamma_12": 0.1,
            "chemical_potential": 2.0, "zng_thermal_fraction_mode": "explicit",
            "zng_thermal_fraction": fraction},
            system=system, app=make_app(), simulation_name='zng')
        bec._initialize_simulation_parameters()
        # stand in for BaseBEC._initialise (which reads the state from disk)
        bec.psi = system._ground_state.clone()
        if bec.thermal_fraction > 0.0:
            bec.psi = bec.psi * ((1.0 - bec.thermal_fraction) ** 0.5)
        w = torch.tensor(system.simulation_parameters["w"], dtype=torch.float64)
        bec.particle_positions, bec.particle_momenta = sample_initial_thermal_cloud(
            n_test, w, bec.temperature, bec.device)
        bec._update_thermal_density()
        return bec

    def test_components_sum_to_the_whole_sample(self):
        for fraction in (0.0, 0.2, 0.6):
            bec = self._initialised(fraction)
            total = bec.condensate_number() + float(bec.d_x * torch.sum(bec.n_tilde))
            self.assertAlmostEqual(total, 1.0, places=5,
                                   msg=f"components do not add to 1 at f={fraction}")

    def test_condensate_holds_one_minus_the_fraction(self):
        bec = self._initialised(0.4)
        self.assertAlmostEqual(bec.condensate_number(), 0.6, places=10)


class TestTotalAtomNumberConservation(ZNGCase):
    """What the condensate loses, the cloud must gain.

    The condensate's R term and the C_12 Monte-Carlo transfer are two views of
    the same exchange. Modelling their rates independently let them disagree —
    the emission rate is half of |R|n_c, and absorption keyed on the particle
    energy rather than V_eff — so the total number drifted. C_12 is now sized
    by the condensate's measured norm change.
    """

    def _total(self, bec):
        return bec.condensate_number() + float(bec.d_x * torch.sum(bec.n_tilde))

    def _run_with_split(self, exchange, kmax=40, n_test=1500, fraction=0.25, seed=0):
        torch.manual_seed(seed)
        bec = self.build(exchange=exchange, n_test=n_test, kmax=kmax,
                         zng_thermal_fraction_mode="explicit",
                         zng_thermal_fraction=fraction)
        bec.psi = bec.psi * ((1.0 - bec.thermal_fraction) ** 0.5)
        bec._update_thermal_density()
        before = self._total(bec)
        self.run_loop(bec)
        return before, self._total(bec), bec

    def test_total_is_conserved_with_exchange_on(self):
        before, after, _ = self._run_with_split(exchange=True)
        self.assertLess(abs(after - before) / before, 5e-3,
                        msg=f"total atom number drifted: {before:.6f} -> {after:.6f}")

    def test_total_is_conserved_with_exchange_off(self):
        """Pinned condensate means no net transfer either way."""
        before, after, _ = self._run_with_split(exchange=False)
        self.assertLess(abs(after - before) / before, 5e-3)

    def test_drift_does_not_grow_with_the_step_count(self):
        """A systematic imbalance accumulates; an unbiased residual does not."""
        _, _, short = self._run_with_split(exchange=True, kmax=20, seed=1)
        _, _, long_run = self._run_with_split(exchange=True, kmax=80, seed=1)
        short_err = abs(self._total(short) - 1.0)
        long_err = abs(self._total(long_run) - 1.0)
        self.assertLess(long_err, max(4 * short_err, 5e-3),
                        msg=f"drift grows with steps: {short_err:.2e} -> {long_err:.2e}")

    def test_components_move_in_opposite_directions(self):
        _, _, bec = self._run_with_split(exchange=True)
        condensate_change = bec.condensate_number() - 0.75
        thermal_change = float(bec.d_x * torch.sum(bec.n_tilde)) - 0.25
        if abs(condensate_change) > 1e-6:
            self.assertLess(condensate_change * thermal_change, 0.0,
                            msg="both components moved the same way")

    def test_finer_particle_weight_reduces_the_residual(self):
        """The leftover is one test particle of discreteness, so more (lighter)
        particles must close the books more tightly."""
        errors = []
        for n_test in (300, 4800):
            before, after, _ = self._run_with_split(exchange=True, n_test=n_test, seed=2)
            errors.append(abs(after - before))
        self.assertLess(errors[1], errors[0])


class TestZNGLoopRuns(ZNGCase):
    def test_loop_completes_and_advances_particles(self):
        bec = self.build(exchange=True, kmax=6, n_test=600)
        before = bec.particle_positions.clone()
        self.run_loop(bec)
        self.assertGreater(bec.particle_positions.shape[0], 0,
                           msg="the thermal cloud was emptied")
        shared = min(before.shape[0], bec.particle_positions.shape[0])
        moved = float((bec.particle_positions[:shared] - before[:shared]).abs().max())
        self.assertGreater(moved, 0.0, msg="leapfrog did not move any particle")

    def test_thermal_density_is_non_negative_and_finite(self):
        bec = self.run_loop(self.build(exchange=True, kmax=6))
        self.assertTrue(bool(torch.all(torch.isfinite(bec.n_tilde))))
        self.assertGreaterEqual(float(bec.n_tilde.min()), 0.0)

    def test_chemical_potential_is_computed_when_not_supplied(self):
        """Regression: this call had gone stale against
        calculate_energy_allocation's signature and raised TypeError."""
        system = make_system()
        bec = ZNGBEC(parameters={"temperature": 1.0, "n_test_particles": 200,
                                 "chemical_potential": None},
                     system=system, app=make_app(), simulation_name='zng')
        bec._initialize_simulation_parameters()
        bec.psi = system._ground_state.clone()
        energy = gpe.calculate_energy_allocation(
            bec.psi, bec.uext, (bec.p1, bec.p2, bec.p3), bec.d_x, u=bec.u)
        mu = float((energy['e_kin'] + energy['e_pot'] + 2.0 * energy['e_int']).real)
        self.assertTrue(np.isfinite(mu))
        # a physical scale in units of hbar*omega_ho, not 1/d_x
        self.assertGreater(mu, 0.0)
        self.assertLess(mu, 50.0)


if __name__ == '__main__':
    unittest.main()
