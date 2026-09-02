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
from unittest.mock import MagicMock, patch

import numpy as np
import torch

sys.path.append('.')

from src.models.base_BEC import BaseBEC
from src.experimental.zng.zng_BEC import ZNGBEC
from src.experimental.zng.zng_library import sample_initial_thermal_cloud
from src.library.gpe_library import GPELibrary as gpe
from src.utils.setup_simulations import get_simulation_combinations


# A minimal configuration file, as the JSON reader would return it. Only the
# keys the setup modules require are present; each parser test overlays the ZNG
# keys it cares about.
_BASE_CONFIG = {
    "Trapping_frequencies": [100.0, 100.0, 10.0],
    "Total_simulation_time": 0.001,
    "dt": 1e-5,
    "snapshots": 5,
    "Potential_type": "harmonic",
    "SwitchOff_time": None,
    "vortex_excitation": False,
    "vortex_charge": [[1]],
    "imprinting_charge": [[1]],
    "vortex_position_x": [[0]],
    "vortex_position_y": [[0]],
    "imprint_position_x": [[0]],
    "imprint_position_y": [[0]],
    "initial_imprint_time": [0],
    "repetitive": 0,
    "imprint_every": [],
    "imprint_times": [],
    "max_imprints": [],
}


def _parse_cartesian_config(**overrides):
    """Run a config through the Cartesian setup module and return its output.

    File I/O is bypassed by patching the JSON reader, so the test supplies the
    configuration dict directly. Raises if the config is rejected, since every
    caller here expects a valid one.
    """
    from unittest.mock import patch
    from src.sims_setup.cartesian_setup import get_simulation_parameters_cartesian

    config = dict(_BASE_CONFIG,
                  Grid_resolution=[32, 32, 32],
                  Grid_negative_limits=[-10, -10, -10],
                  Grid_positive_limits=[10, 10, 10],
                  **overrides)
    with patch("src.sims_setup.cartesian_setup._load_json_from_cwd",
               return_value=config):
        params, message = get_simulation_parameters_cartesian("dummy.json")
    if params is None:
        raise AssertionError(f"config rejected by the Cartesian parser: {message}")
    return params


def _parse_cylindrical_config(**overrides):
    """Run a config through the cylindrical setup module and return its output."""
    from unittest.mock import patch
    from src.sims_setup.cylindrical_setup import get_simulation_parameters_cylindrical

    config = dict(_BASE_CONFIG,
                  Grid_resolution=[32, 8, 32],
                  r_max=10.0, z_min=-10.0, z_max=10.0,
                  Trapping_frequencies=[100.0, 10.0],
                  **overrides)
    with patch("src.sims_setup.cylindrical_setup._load_json_from_cwd",
               return_value=config):
        params, message = get_simulation_parameters_cylindrical("dummy.json")
    if params is None:
        raise AssertionError(f"config rejected by the cylindrical parser: {message}")
    return params


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
        """The model reads the switch out of its own parameters dict."""
        self.assertTrue(self.build(exchange=True).condensate_exchange)
        self.assertFalse(self.build(exchange=False).condensate_exchange)

    def test_cartesian_config_parses_the_zng_settings(self):
        """The Cartesian setup module carries the ZNG keys into
        ``system.simulation_parameters``.

        This drives the parser rather than grepping its source: an earlier
        version asserted only that the string ``zng_condensate_exchange``
        appeared somewhere in the function body, which stayed green no matter
        what the function actually produced.
        """
        params = _parse_cartesian_config(zng_condensate_exchange=True,
                                         zng_thermal_fraction_mode="explicit",
                                         zng_thermal_fraction=0.3)
        self.assertTrue(params["zng_condensate_exchange"])
        self.assertEqual(params["zng_thermal_fraction_mode"], "explicit")
        self.assertAlmostEqual(params["zng_thermal_fraction"], 0.3)

    def test_cylindrical_config_parses_the_zng_settings(self):
        """The cylindrical setup module carries the same keys through."""
        params = _parse_cylindrical_config(zng_condensate_exchange=True,
                                           zng_thermal_fraction_mode="explicit",
                                           zng_thermal_fraction=0.3)
        self.assertTrue(params["zng_condensate_exchange"])
        self.assertEqual(params["zng_thermal_fraction_mode"], "explicit")
        self.assertAlmostEqual(params["zng_thermal_fraction"], 0.3)

    def test_config_defaults_match_the_model_defaults(self):
        """A config that says nothing about ZNG parses to the model's own
        defaults, so an omitted key and an explicit default agree."""
        params = _parse_cartesian_config()
        self.assertFalse(params["zng_condensate_exchange"])
        self.assertEqual(params["zng_thermal_fraction_mode"], "temperature")
        self.assertIsNone(params["zng_thermal_fraction"])

    @unittest.expectedFailure
    def test_zng_settings_reach_the_per_simulation_parameters(self):
        """Known gap: the parsed ZNG keys never reach the model.

        ``Simulations.run_simulations`` hands each model the dict built by
        ``get_simulation_combinations``, and its ``finite_temp_params`` copies
        only ``model_type``, ``temperature``, ``damping_coefficient``,
        ``n_test_particles``, ``gamma_12``, ``chemical_potential`` and
        ``enable_c22``. The three ``zng_*`` keys are parsed into
        ``system.simulation_parameters`` and then dropped, so setting them in a
        configuration file has no effect and ``ZNGBEC`` always falls back to its
        defaults.

        Marked as an expected failure so the gap is recorded rather than
        asserted away. Adding the keys to ``finite_temp_params`` fixes it; this
        test then reports an unexpected success and should lose the decorator.
        """
        params = _parse_cartesian_config(zng_condensate_exchange=True,
                                         vortex_excitation=1)
        # The builder prints and creates a folder per simulation; setUp has
        # already moved us into a scratch directory.
        with contextlib.redirect_stdout(io.StringIO()):
            combinations = get_simulation_combinations(params)
        _, per_simulation = combinations[0]
        self.assertIn("zng_condensate_exchange", per_simulation)

    def test_other_zng_settings_still_load(self):
        """The settings that do reach the model are read off the parameters."""
        bec = self.build(exchange=True, n_test=250, gamma_12=0.25, enable_c22=True)
        self.assertEqual(bec.n_test, 250)
        self.assertEqual(bec.gamma_12, 0.25)
        self.assertTrue(bec.enable_c22)


class TestZNGCondensateNumber(ZNGCase):
    def test_pinned_mode_holds_the_condensate_number(self):
        """With exchange off the condensate number is 1.0 to machine precision.

        The reconciliation step rescales psi back to the number it started the step
        with, so ten steps must leave the norm untouched rather than merely close.
        """
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
        """condensate_number() is the discretised integral of |psi|^2, cell volume
        included.

        Without the d_x weight the figure would scale with the grid spacing, and
        every conservation test in this module compares it against the thermal
        integral, which does carry its volume element.
        """
        bec = self.build(exchange=True)
        expected = float(bec.d_x * torch.sum(torch.abs(bec.psi) ** 2))
        self.assertAlmostEqual(bec.condensate_number(), expected, places=12)

    def test_state_stays_finite_in_both_modes(self):
        """Neither mode drives psi to NaN or infinity over ten steps.

        The source term R enters the split step as a real exponential, so a sign
        slip there shows up as an overflow rather than as a wrong-looking number.
        """
        for exchange in (False, True):
            bec = self.run_loop(self.build(exchange=exchange, kmax=10))
            self.assertTrue(bool(torch.all(torch.isfinite(torch.abs(bec.psi)))),
                            msg=f"psi went non-finite with exchange={exchange}")

    def test_the_two_modes_differ(self):
        """Turning exchange on changes the outcome, from the same random seed.

        Both runs are seeded identically, so the difference cannot come from the
        Monte Carlo sampling: it is the flag alone. Without this, a flag that was
        read but never acted on would still satisfy every other test here.
        """
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
        """Above T_c the whole sample is thermal, so the fraction is capped at 1.

        The ideal-Bose expression T^3 zeta(3) / N grows without bound; the cap is
        what keeps the condensate scaling factor sqrt(1 - f) real.
        """
        bec = self.build(temperature=1e6, zng_thermal_fraction_mode="temperature")
        self.assertEqual(bec.thermal_fraction, 1.0)

    def test_temperature_convention_is_monotonic(self):
        """A hotter cloud holds a larger thermal fraction.

        The comparison is strict: ``sorted()`` alone would also accept a
        constant sequence, i.e. a fraction that ignored the temperature.
        """
        fractions = [self.build(temperature=kT,
                                zng_thermal_fraction_mode="temperature").thermal_fraction
                     for kT in (1.0, 10.0, 30.0)]
        for cooler, hotter in zip(fractions, fractions[1:]):
            self.assertLess(cooler, hotter, msg=f"not increasing: {fractions}")

    def test_explicit_convention_uses_the_configured_value(self):
        """In explicit mode the configured fraction is used verbatim."""
        bec = self.build(zng_thermal_fraction_mode="explicit",
                         zng_thermal_fraction=0.35)
        self.assertAlmostEqual(bec.thermal_fraction, 0.35, places=12)

    def test_explicit_convention_ignores_the_temperature(self):
        """In explicit mode the temperature does not enter the fraction at all.

        The two runs differ by a factor of 500 in temperature and must still agree
        exactly, which is the point of the mode: it lets f be swept independently
        of T, or matched to a measured condensate fraction.
        """
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
        """A fraction outside [0, 1] is rejected rather than silently clamped.

        A negative f would make the condensate scaling factor sqrt(1 - f) exceed 1
        and f > 1 would make it imaginary, so neither can be allowed through.
        """
        for bad in (-0.1, 1.5):
            with self.assertRaises(ValueError):
                self.build(zng_thermal_fraction_mode="explicit",
                           zng_thermal_fraction=bad)

    def test_unknown_convention_is_rejected(self):
        """An unrecognised mode name raises instead of falling back to a default."""
        with self.assertRaises(ValueError):
            self.build(zng_thermal_fraction_mode="whatever")

    def test_default_convention_is_temperature(self):
        """A config that says nothing gets the temperature convention."""
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
        """One test particle stands for thermal_fraction / n_test of the sample.

        This is the weight that puts the thermal density on the condensate's own
        normalisation; the two tests above check what it does to the integral, and
        this one pins the number itself.
        """
        bec = self.build(zng_thermal_fraction_mode="explicit",
                         zng_thermal_fraction=0.5, n_test=1000)
        self.assertAlmostEqual(bec.particle_weight, 0.5 / 1000, places=15)

    def test_zero_fraction_leaves_an_empty_cloud(self):
        """At f = 0 the deposited thermal density is exactly zero everywhere.

        The test particles are still sampled and still move; they simply carry no
        weight, so the mean-field terms that mix the components vanish and the run
        reduces to the zero-temperature GPE.
        """
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
        """Condensate plus cloud integrate to 1 for several splits.

        The ground state arrives holding every atom; the split scales it by
        sqrt(1 - f) and gives the cloud the remaining f. The tolerance is loose
        enough to absorb the Monte Carlo noise in the deposited density, which is
        what the 500 test particles produce.
        """
        for fraction in (0.0, 0.2, 0.6):
            bec = self._initialised(fraction)
            total = bec.condensate_number() + float(bec.d_x * torch.sum(bec.n_tilde))
            self.assertAlmostEqual(total, 1.0, places=5,
                                   msg=f"components do not add to 1 at f={fraction}")

    def test_condensate_holds_one_minus_the_fraction(self):
        """At f = 0.4 the condensate is left holding exactly 0.6 of the sample.

        Unlike the sum above, this side of the split is deterministic -- a uniform
        rescaling of psi -- so it can be asserted to ten places.
        """
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
        """Running with exchange on leaves the total unchanged to within 0.5%.

        The residual is the single test particle that discreteness allows per step
        (see _stochastic_round), not a systematic imbalance; the two tests below
        separate those two possibilities.
        """
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
        """Atoms the condensate gains are atoms the cloud lost, and vice versa.

        The two changes are asserted to have opposite signs. The condensate
        movement is checked first: without that precondition the sign test
        would pass vacuously on a run where nothing was transferred at all,
        which is the failure mode it exists to catch.
        """
        _, _, bec = self._run_with_split(exchange=True)
        condensate_change = bec.condensate_number() - 0.75
        thermal_change = float(bec.d_x * torch.sum(bec.n_tilde)) - 0.25
        self.assertGreater(abs(condensate_change), 1e-6,
                           msg="the condensate did not exchange any atoms, so "
                               "the sign test below would prove nothing")
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
        """The loop runs to completion, keeps a non-empty cloud and moves the particles.

        C_12 adds and removes particles, so the arrays are compared only over the
        indices both runs share. A leapfrog step that silently produced zero force
        would leave every position untouched, which is what the final assertion
        rules out.
        """
        bec = self.build(exchange=True, kmax=6, n_test=600)
        before = bec.particle_positions.clone()
        self.run_loop(bec)
        self.assertGreater(bec.particle_positions.shape[0], 0,
                           msg="the thermal cloud was emptied")
        shared = min(before.shape[0], bec.particle_positions.shape[0])
        moved = float((bec.particle_positions[:shared] - before[:shared]).abs().max())
        self.assertGreater(moved, 0.0, msg="leapfrog did not move any particle")

    def test_thermal_density_is_non_negative_and_finite(self):
        """The deposited density is finite and never negative.

        Cloud-In-Cell weights are products of numbers in [0, 1], so a negative cell
        means the deposition indices or weights have gone wrong.
        """
        bec = self.run_loop(self.build(exchange=True, kmax=6))
        self.assertTrue(bool(torch.all(torch.isfinite(bec.n_tilde))))
        self.assertGreaterEqual(float(bec.n_tilde.min()), 0.0)

    def test_chemical_potential_is_computed_when_not_supplied(self):
        """``_initialise`` fills in mu from the ground state when the config
        leaves it out.

        Regression: the call inside ``_initialise`` had gone stale against
        ``calculate_energy_allocation``'s signature and raised TypeError. The
        test therefore has to drive ``ZNGBEC._initialise`` itself — an earlier
        version recomputed mu inline with the same formula, which exercised the
        library rather than the model and could not have caught the defect.

        Only ``BaseBEC._initialise`` is stubbed out, since that is the part
        that reads the state from a file on disk.
        """
        system = make_system()
        bec = ZNGBEC(parameters={"temperature": 1.0, "n_test_particles": 200,
                                 "chemical_potential": None},
                     system=system, app=make_app(), simulation_name='zng')
        bec._initialize_simulation_parameters()

        def load_ground_state(self):
            self.psi = system._ground_state.clone()

        with patch.object(BaseBEC, '_initialise', load_ground_state):
            bec._initialise()

        self.assertIsNotNone(bec.mu, msg="mu was left unset by _initialise")
        self.assertTrue(np.isfinite(bec.mu))
        # a physical scale in units of hbar*omega_ho, not 1/d_x
        self.assertGreater(bec.mu, 0.0)
        self.assertLess(bec.mu, 50.0)

    def test_configured_chemical_potential_is_left_alone(self):
        """A mu given in the config is used as-is, not recomputed."""
        system = make_system()
        bec = ZNGBEC(parameters={"temperature": 1.0, "n_test_particles": 200,
                                 "chemical_potential": 3.25},
                     system=system, app=make_app(), simulation_name='zng')
        bec._initialize_simulation_parameters()

        def load_ground_state(self):
            self.psi = system._ground_state.clone()

        with patch.object(BaseBEC, '_initialise', load_ground_state):
            bec._initialise()

        self.assertAlmostEqual(bec.mu, 3.25, places=12)


if __name__ == '__main__':
    unittest.main()
