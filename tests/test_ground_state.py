import contextlib
import io
import os
import sys
import tempfile
import types
import unittest

import numpy as np
import torch

sys.path.append('.')

from src.library.ground_state import GroundState
from src.library.ground_state_cylindrical import CylindricalGroundState
from src.library.gpe_cylindrical_library import GPECylindricalLibrary as cyl
from src.library.gpe_library import GPELibrary as gpe
from src.utils.read_write_utils import write_psi


class TestGroundState(unittest.TestCase):
    """Focused regression tests for the GroundState steepest-descent step."""

    def _make_energy_regression_fixture(self):
        """Build a tiny deterministic fixture for future hand-checked regressions."""
        device = torch.device('cpu')
        psi = torch.tensor(
            [
                [[1.0 + 0.5j, 0.5 - 0.25j], [0.25 + 0.75j, -0.5 + 0.125j]],
                [[-0.75 + 0.5j, 0.625 + 0.25j], [0.125 - 0.5j, 0.875 + 0.0j]],
            ],
            dtype=torch.cdouble,
            device=device,
        )
        d_x = 0.25
        psi = psi / torch.sqrt(d_x * torch.sum(torch.abs(psi) ** 2))
        p_sq = torch.tensor(
            [
                [[0.0, 0.2], [0.4, 0.6]],
                [[0.8, 1.0], [1.2, 1.4]],
            ],
            dtype=torch.float64,
            device=device,
        )
        uext = torch.tensor(
            [
                [[0.1, 0.15], [0.2, 0.25]],
                [[0.3, 0.35], [0.4, 0.45]],
            ],
            dtype=torch.float64,
            device=device,
        ).to(torch.cdouble)
        return psi, p_sq, uext, d_x, 0.2

    def test_steepest_descent_returns_tensor_diagnostics_and_normalizes(self):
        """The solver should keep diagnostics as tensors and renormalize the state."""
        device = torch.device('cpu')
        psi = torch.ones((4, 4, 4), dtype=torch.cdouble, device=device)
        p_sq = torch.zeros((4, 4, 4), dtype=torch.float64, device=device)
        uext = torch.zeros((4, 4, 4), dtype=torch.cdouble, device=device)
        d_x = 0.5

        updated_psi, energy, tol, mu = GroundState.steepest_descent(
            psi,
            dtau=0.01,
            p_sq=p_sq,
            uext=uext,
            d_x=d_x,
            u=0.5,
        )

        self.assertTrue(torch.is_tensor(updated_psi))
        self.assertTrue(torch.is_tensor(energy))
        self.assertTrue(torch.is_tensor(tol))
        self.assertTrue(torch.is_tensor(mu))
        self.assertEqual(updated_psi.device.type, device.type)
        self.assertEqual(energy.device.type, device.type)
        self.assertEqual(tol.device.type, device.type)
        self.assertEqual(mu.device.type, device.type)
        self.assertEqual(energy.ndim, 0)
        self.assertEqual(tol.ndim, 0)
        self.assertEqual(mu.ndim, 0)
        self.assertGreaterEqual(tol.real.item(), 0.0)

        norm_value = d_x * torch.sum(torch.abs(updated_psi) ** 2)
        self.assertAlmostEqual(norm_value.item(), 1.0, places=12)

    @unittest.skip('Fill in the hand-computed reference values for energy, mu, and tol.')
    def test_steepest_descent_matches_hand_computed_reference(self):
        """Scaffold: pin one steepest-descent step against reference values.

        Skipped because the three expected values below are still None, so the
        assertions would raise TypeError rather than fail informatively.

        To finish it, compute energy, tol and mu for the fixture built by
        _make_energy_regression_fixture from an independent implementation --
        not by recording what the current code returns, which would pin the
        behaviour rather than check it -- and drop the skip decorator. The
        surrounding tests already cover shapes, dtypes, devices and the norm;
        what is missing here is the exact arithmetic of the scalars.
        """
        psi, p_sq, uext, d_x, interaction_strength = self._make_energy_regression_fixture()

        _, energy, tol, mu = GroundState.steepest_descent(
            psi,
            dtau=0.01,
            p_sq=p_sq,
            uext=uext,
            d_x=d_x,
            u=interaction_strength,
        )

        expected_energy = None
        expected_tol = None
        expected_mu = None

        self.assertAlmostEqual(energy.real.item(), expected_energy, places=12)
        self.assertAlmostEqual(tol.real.item(), expected_tol, places=12)
        self.assertAlmostEqual(mu.real.item(), expected_mu, places=12)

    def test_steepest_descent_with_potential_and_kinetic_terms_stays_finite(self):
        """A nontrivial input should yield finite diagnostics and a changed state."""
        device = torch.device('cpu')
        base = torch.arange(1, 28, dtype=torch.float64, device=device).reshape(3, 3, 3)
        psi = (base + 0.1j * torch.flip(base, dims=(0,))).to(torch.cdouble)
        psi = psi / torch.sqrt(torch.sum(torch.abs(psi) ** 2))
        p_sq = torch.linspace(0.0, 2.6, steps=27, dtype=torch.float64, device=device).reshape(3, 3, 3)
        uext = torch.linspace(0.2, 1.4, steps=27, dtype=torch.float64, device=device).reshape(3, 3, 3).to(torch.cdouble)
        d_x = 1.0

        updated_psi, energy, tol, mu = GroundState.steepest_descent(
            psi,
            dtau=0.005,
            p_sq=p_sq,
            uext=uext,
            d_x=d_x,
            u=0.3,
        )

        self.assertEqual(updated_psi.shape, psi.shape)
        self.assertTrue(torch.all(torch.isfinite(updated_psi.real)))
        self.assertTrue(torch.all(torch.isfinite(updated_psi.imag)))
        self.assertTrue(torch.isfinite(energy.real))
        self.assertTrue(torch.isfinite(tol.real))
        self.assertTrue(torch.isfinite(mu.real))
        self.assertGreater(tol.real.item(), 0.0)
        self.assertGreater(mu.real.item(), 0.0)
        self.assertFalse(torch.allclose(updated_psi, psi))

        norm_value = d_x * torch.sum(torch.abs(updated_psi) ** 2)
        self.assertAlmostEqual(norm_value.item(), 1.0, places=12)


class TestCylindricalGroundState(unittest.TestCase):
    """Focused regression tests for CylindricalGroundState steepest-descent step."""

    def _make_cylindrical_fixture(self, n_r=4, n_phi=4, n_z=4):
        """Build a small deterministic cylindrical fixture."""
        device = torch.device('cpu')
        r_max, z_min, z_max = 4.0, -2.0, 2.0

        r, _, _, kz, m_modes, dr, dphi, dz, _ = cyl.init_grid(
            r_max, z_min, z_max, n_r, n_phi, n_z, device
        )
        eigvecs_dict, eigvals_dict = cyl.build_radial_operators(r, dr, m_modes, device)

        r_w = r.reshape(-1, 1, 1)
        psi = torch.exp(-r_w ** 2 / 2.0).expand(n_r, n_phi, n_z).clone().to(torch.cdouble)
        psi = cyl.normalize(psi, r, dr, dphi, dz)
        uext = torch.zeros((n_r, n_phi, n_z), dtype=torch.cdouble, device=device)

        return psi, kz, m_modes, r, eigvecs_dict, eigvals_dict, uext, dr, dphi, dz, device

    def test_apply_kinetic_returns_correct_shape_and_finite(self):
        """apply_kinetic should return a finite tensor of the same shape as input."""
        psi, kz, m_modes, r, eigvecs_dict, eigvals_dict, _, _, _, _, device = (
            self._make_cylindrical_fixture()
        )

        result = CylindricalGroundState.apply_kinetic(
            psi, kz, m_modes, r, eigvecs_dict, eigvals_dict
        )

        self.assertTrue(torch.is_tensor(result))
        self.assertEqual(result.shape, psi.shape)
        self.assertEqual(result.device.type, device.type)
        self.assertTrue(torch.all(torch.isfinite(result.real)))
        self.assertTrue(torch.all(torch.isfinite(result.imag)))

    def test_steepest_descent_returns_tensor_diagnostics_and_normalizes(self):
        """The cylindrical solver should keep diagnostics as tensors and renormalize with cylindrical norm."""
        psi, kz, m_modes, r, eigvecs_dict, eigvals_dict, uext, dr, dphi, dz, device = (
            self._make_cylindrical_fixture()
        )

        updated_psi, energy, tol, mu = CylindricalGroundState.steepest_descent(
            psi,
            dtau=0.01,
            kz=kz,
            m_modes=m_modes,
            r=r,
            eigvecs_dict=eigvecs_dict,
            eigvals_dict=eigvals_dict,
            uext=uext,
            dr=dr,
            dphi=dphi,
            dz=dz,
            u=0.5,
        )

        self.assertTrue(torch.is_tensor(updated_psi))
        self.assertTrue(torch.is_tensor(energy))
        self.assertTrue(torch.is_tensor(tol))
        self.assertTrue(torch.is_tensor(mu))
        self.assertEqual(updated_psi.device.type, device.type)
        self.assertEqual(energy.device.type, device.type)
        self.assertEqual(tol.device.type, device.type)
        self.assertEqual(mu.device.type, device.type)
        self.assertEqual(energy.ndim, 0)
        self.assertEqual(tol.ndim, 0)
        self.assertEqual(mu.ndim, 0)
        self.assertGreaterEqual(tol.real.item(), 0.0)

        # Cylindrical normalization: ∫|ψ|² r dr dφ dz = 1
        r_w = r.reshape(-1, 1, 1)
        norm_value = torch.sum(torch.abs(updated_psi) ** 2 * r_w) * (dr * dphi * dz)
        self.assertAlmostEqual(norm_value.item(), 1.0, places=12)

    def test_steepest_descent_with_potential_and_kinetic_terms_stays_finite(self):
        """A nontrivial cylindrical input should yield finite diagnostics and a changed state."""
        device = torch.device('cpu')
        n_r, n_phi, n_z = 3, 4, 4

        r, _, _, kz, m_modes, dr, dphi, dz, _ = cyl.init_grid(
            3.0, -1.5, 1.5, n_r, n_phi, n_z, device
        )
        eigvecs_dict, eigvals_dict = cyl.build_radial_operators(r, dr, m_modes, device)

        base = torch.arange(1, n_r * n_phi * n_z + 1, dtype=torch.float64, device=device).reshape(
            n_r, n_phi, n_z
        )
        psi = (base + 0.1j * torch.flip(base, dims=(0,))).to(torch.cdouble)
        psi = cyl.normalize(psi, r, dr, dphi, dz)
        uext = (
            torch.linspace(0.2, 1.4, steps=n_r * n_phi * n_z, dtype=torch.float64, device=device)
            .reshape(n_r, n_phi, n_z)
            .to(torch.cdouble)
        )

        updated_psi, energy, tol, mu = CylindricalGroundState.steepest_descent(
            psi,
            dtau=0.005,
            kz=kz,
            m_modes=m_modes,
            r=r,
            eigvecs_dict=eigvecs_dict,
            eigvals_dict=eigvals_dict,
            uext=uext,
            dr=dr,
            dphi=dphi,
            dz=dz,
            u=0.3,
        )

        self.assertEqual(updated_psi.shape, psi.shape)
        self.assertTrue(torch.all(torch.isfinite(updated_psi.real)))
        self.assertTrue(torch.all(torch.isfinite(updated_psi.imag)))
        self.assertTrue(torch.isfinite(energy.real))
        self.assertTrue(torch.isfinite(tol.real))
        self.assertTrue(torch.isfinite(mu.real))
        self.assertGreater(tol.real.item(), 0.0)
        self.assertGreater(mu.real.item(), 0.0)
        self.assertFalse(torch.allclose(updated_psi, psi))

        r_w = r.reshape(-1, 1, 1)
        norm_value = torch.sum(torch.abs(updated_psi) ** 2 * r_w) * (dr * dphi * dz)
        self.assertAlmostEqual(norm_value.item(), 1.0, places=12)

    @unittest.skip('Fill in the hand-computed reference values for energy, mu, and tol.')
    def test_steepest_descent_matches_hand_computed_reference(self):
        """Scaffold: the cylindrical counterpart of the Cartesian reference test.

        Skipped for the same reason, and needs the same work: reference values
        for energy, tol and mu computed independently of this implementation,
        here for the fixture from _make_cylindrical_fixture, with its r dr dphi
        dz volume element and its per-mode radial operators.
        """
        psi, kz, m_modes, r, eigvecs_dict, eigvals_dict, uext, dr, dphi, dz, _ = (
            self._make_cylindrical_fixture()
        )

        _, energy, tol, mu = CylindricalGroundState.steepest_descent(
            psi,
            dtau=0.01,
            kz=kz,
            m_modes=m_modes,
            r=r,
            eigvecs_dict=eigvecs_dict,
            eigvals_dict=eigvals_dict,
            uext=uext,
            dr=dr,
            dphi=dphi,
            dz=dz,
            u=0.2,
        )

        expected_energy = None
        expected_tol = None
        expected_mu = None

        self.assertAlmostEqual(energy.real.item(), expected_energy, places=12)
        self.assertAlmostEqual(tol.real.item(), expected_tol, places=12)
        self.assertAlmostEqual(mu.real.item(), expected_mu, places=12)


# ----------------------------------------------------------------------
# Convergence behaviour of the full solvers
# ----------------------------------------------------------------------

class _StubSystem:
    """Minimal stand-in for the System object the solvers read from."""

    def __init__(self, params, potential):
        self.simulation_parameters = params
        self.uext = types.SimpleNamespace(potential=potential)


class TestCartesianConvergence(unittest.TestCase):
    """End-to-end behaviour of GroundState.find_ground_state."""

    @classmethod
    def setUpClass(cls):
        cls.n = 24
        cls.L = 10.0
        cls.dx = np.array([cls.L / cls.n] * 3)
        cls.d_x = float(np.prod(cls.dx))
        cls.x_min = np.array([-cls.L / 2] * 3)
        (cls.x1, cls.x2, cls.x3, cls.p1, cls.p2, cls.p3, cls.p_sq,
         (cls.gx, cls.gy, cls.gz), _) = gpe.init_grid(
            cls.x_min, cls.dx, [2 * np.pi / cls.L] * 3, cls.n, cls.n, cls.n, 'cpu')
        cls.harmonic = 0.5 * (cls.gx ** 2 + cls.gy ** 2 + cls.gz ** 2)

    def _system(self, potential, **overrides):
        params = {
            "Grid_resolution": [self.n, self.n, self.n],
            "d_x": self.d_x, "dx": self.dx,
            "dp": [2 * np.pi / self.L] * 3,
            "x_min": self.x_min, "a_ho": 1e-6, "u": 0.0,
        }
        params.update(overrides)
        return _StubSystem(params, potential)

    def _solve(self, system, **kwargs):
        target = os.path.join(tempfile.gettempdir(), "_gs_test.dat")
        try:
            with contextlib.redirect_stdout(io.StringIO()) as captured:
                psi = GroundState.find_ground_state({}, system, target, 'cpu', **kwargs)
            return psi, captured.getvalue()
        finally:
            if os.path.exists(target):
                os.remove(target)

    def test_converges_to_the_analytic_ground_state_energy(self):
        """The solver reaches E = 3/2 hbar*omega_ho for a non-interacting isotropic trap.

        That is the exact 3-D harmonic-oscillator ground-state energy, so this is a
        value check rather than a self-consistency one; three decimal places is
        what the 32-point grid supports.
        """
        psi, _ = self._solve(self._system(self.harmonic))
        energies = gpe.calculate_energy_allocation(
            psi, self.harmonic, (self.p1, self.p2, self.p3), self.d_x, u=0.0)
        self.assertAlmostEqual(float(energies['E_total'].real), 1.5, places=3)

    def test_result_is_normalised(self):
        """The returned state is normalised with the volume element included.

        Every energy computed from it divides by this norm implicitly, so a state
        that came back unnormalised would shift all of them together.
        """
        psi, _ = self._solve(self._system(self.harmonic))
        self.assertAlmostEqual(float(self.d_x * torch.sum(torch.abs(psi) ** 2)), 1.0, places=10)

    def test_negative_chemical_potential_still_converges(self):
        """Regression: taking |μ| instead of Re(μ) inverted the descent for any
        trap with a negative energy offset, so the residual grew without bound."""
        offset = self.harmonic - 8.0
        psi, _ = self._solve(self._system(offset))
        mu = gpe.calculate_chemical_potential(
            psi, offset, 0.0, (self.p1, self.p2, self.p3), self.d_x)
        self.assertLess(mu, 0.0, msg="fixture should produce a negative mu")

        # A constant offset cannot change the ground state, only its energy.
        reference, _ = self._solve(self._system(self.harmonic))
        self.assertTrue(torch.allclose(torch.abs(psi), torch.abs(reference), atol=1e-3))
        self.assertAlmostEqual(mu + 8.0,
                               gpe.calculate_chemical_potential(
                                   reference, self.harmonic, 0.0,
                                   (self.p1, self.p2, self.p3), self.d_x),
                               places=3)

    def test_iteration_cap_stops_a_run_that_cannot_converge(self):
        """Without a cap this loop had no upper bound on its runtime.

        A strongly interacting state needs far more than a handful of descent
        steps, so a tiny cap is guaranteed to bite.
        """
        psi, log = self._solve(self._system(self.harmonic, u=200.0), max_iterations=3)
        self.assertIn("did not converge", log)
        self.assertTrue(bool(torch.all(torch.isfinite(torch.abs(psi)))))

    def test_interaction_strength_is_read_from_the_configuration(self):
        """u comes from the config rather than being re-derived from CONSTANTS."""
        weak, _ = self._solve(self._system(self.harmonic, u=0.0))
        strong, _ = self._solve(self._system(self.harmonic, u=50.0))
        peak_weak = float(torch.abs(weak).max())
        peak_strong = float(torch.abs(strong).max())
        self.assertLess(peak_strong, peak_weak,
                        msg="repulsion should flatten the density profile")

    def test_thomas_fermi_seed_tolerates_a_complex_potential(self):
        """An absorbing potential must not break the seed's comparison."""
        absorbing = self.harmonic.to(torch.cdouble) - 0.5j * torch.ones_like(self.harmonic)
        psi, _ = self._solve(self._system(absorbing))
        self.assertTrue(bool(torch.all(torch.isfinite(torch.abs(psi)))))

    def test_seed_never_evaluates_a_negative_square_root(self):
        """Clamping keeps NaNs out of the initial state entirely."""
        deep = self.harmonic * 50.0
        psi, _ = self._solve(self._system(deep), max_iterations=2)
        self.assertFalse(bool(torch.any(torch.isnan(torch.abs(psi)))))

    def test_weak_interactions_do_not_produce_an_empty_seed(self):
        """Regression: when mu_TF falls below the trap minimum the Thomas-Fermi
        profile is identically zero, and normalising it yields a field of NaNs
        that the descent can never leave. u = 0 is the extreme case."""
        psi, _ = self._solve(self._system(self.harmonic, u=0.0))
        self.assertFalse(bool(torch.any(torch.isnan(torch.abs(psi)))))
        self.assertAlmostEqual(float(self.d_x * torch.sum(torch.abs(psi) ** 2)), 1.0, places=10)

    def test_non_interacting_seed_recovers_the_gaussian_ground_state(self):
        """The fallback seed is exact for u = 0, so E must come out at 1.5."""
        psi, _ = self._solve(self._system(self.harmonic, u=0.0))
        energies = gpe.calculate_energy_allocation(
            psi, self.harmonic, (self.p1, self.p2, self.p3), self.d_x, u=0.0)
        self.assertAlmostEqual(float(energies['E_total'].real), 1.5, places=2)

    def test_a_diverging_run_raises_instead_of_spinning(self):
        """NaN compares False against every threshold, so the loop needs an
        explicit check or it silently runs to the iteration cap."""
        exploding = self.harmonic.clone()
        exploding[0, 0, 0] = float('nan')
        with self.assertRaises(FloatingPointError):
            self._solve(self._system(exploding, u=20.0))


class TestGroundStateSerialisation(unittest.TestCase):
    """write_psi / read_ground_state must be an exact round trip."""

    def setUp(self):
        self.path = os.path.join(tempfile.gettempdir(), "_gs_roundtrip.dat")

    def tearDown(self):
        if os.path.exists(self.path):
            os.remove(self.path)

    def test_round_trip_is_bit_exact(self):
        """Random complex data survives write then read unchanged.

        The writer formats with %.17g, which round-trips float64 exactly, so the
        comparison is torch.equal rather than allclose: a state reloaded from disk
        is meant to continue a run, not approximate it.
        """
        torch.manual_seed(0)
        psi = torch.randn(4, 3, 5, dtype=torch.cdouble)
        write_psi(self.path, psi, 4, 3, 5)
        self.assertTrue(torch.equal(GroundState.read_ground_state(self.path, 4, 3, 5), psi))

    def test_row_major_ordering_is_preserved(self):
        """The flat file is written and read back in the same row-major order.

        The values are 0..23 on a (2, 3, 4) grid, so any transposition of the axes
        shows up as a mismatch; identical values everywhere would hide it.
        """
        psi = torch.arange(24, dtype=torch.float64).reshape(2, 3, 4).to(torch.cdouble)
        write_psi(self.path, psi, 2, 3, 4)
        self.assertTrue(torch.equal(GroundState.read_ground_state(self.path, 2, 3, 4), psi))

    def test_reading_with_the_wrong_grid_raises(self):
        """Reading a file into a grid of the wrong size raises rather than reshaping.

        The ground-state file name encodes the resolution, so a mismatch means the
        cached state belongs to a different run and must not be reused.
        """
        psi = torch.ones(2, 2, 2, dtype=torch.cdouble)
        write_psi(self.path, psi, 2, 2, 2)
        with self.assertRaises(ValueError):
            GroundState.read_ground_state(self.path, 3, 3, 3)

    def test_writing_a_mismatched_grid_raises(self):
        """Writing a state whose shape disagrees with the declared grid raises."""
        with self.assertRaises(ValueError):
            write_psi(self.path, torch.ones(2, 2, 2, dtype=torch.cdouble), 3, 3, 3)

    def test_cylindrical_reader_shares_the_format(self):
        """The cylindrical reader consumes the same file format as the Cartesian one.

        Both geometries share write_psi, so a divergence in the reader would only
        show up when a cylindrical run tried to reuse a cached state.
        """
        torch.manual_seed(1)
        psi = torch.randn(3, 4, 2, dtype=torch.cdouble)
        write_psi(self.path, psi, 3, 4, 2)
        loaded = CylindricalGroundState.read_ground_state(self.path, 3, 4, 2)
        self.assertTrue(torch.equal(loaded, psi))

    def test_cylindrical_reader_validates_the_grid(self):
        """The cylindrical reader rejects a size mismatch too."""
        write_psi(self.path, torch.ones(2, 2, 2, dtype=torch.cdouble), 2, 2, 2)
        with self.assertRaises(ValueError):
            CylindricalGroundState.read_ground_state(self.path, 4, 4, 4)


class TestCylindricalKineticOperator(unittest.TestCase):
    """apply_kinetic is the cylindrical replacement for IFFT(p²/2 · FFT(ψ))."""

    @classmethod
    def setUpClass(cls):
        cls.n_r, cls.n_phi, cls.n_z = 32, 8, 16
        (cls.r, cls.phi, cls.z, cls.kz, cls.m_modes,
         cls.dr, cls.dphi, cls.dz, cls.space_grid) = cyl.init_grid(
            5.0, -4.0, 4.0, cls.n_r, cls.n_phi, cls.n_z, 'cpu')
        cls.eigvecs, cls.eigvals = cyl.build_radial_operators(
            cls.r, cls.dr, cls.m_modes, 'cpu')
        cls.r_w = cls.r.reshape(-1, 1, 1)
        cls.dV = cls.dr * cls.dphi * cls.dz

    def _random(self, seed=0):
        torch.manual_seed(seed)
        psi = torch.randn(self.n_r, self.n_phi, self.n_z, dtype=torch.cdouble)
        return cyl.normalize(psi, self.r, self.dr, self.dphi, self.dz)

    def test_matches_a_per_phi_reference_implementation(self):
        """The vectorised kinetic operator agrees with a loop over azimuthal modes.

        The reference builds the same operator the obvious way -- one eigenbasis
        per |m|, the sqrt(r) symmetrisation applied by hand, the axial part through
        an explicit FFT pair -- so a batching or stacking error in the fast path
        shows up as a difference at the 1e-12 level.
        """
        psi = self._random()
        sqrt_r = torch.sqrt(self.r)
        psi_m = torch.fft.fft(psi, dim=1, norm="ortho")
        expected = torch.zeros_like(psi_m)
        for phi_idx in range(self.n_phi):
            m_abs = int(abs(self.m_modes[phi_idx].item()))
            V = self.eigvecs[m_abs].to(torch.cdouble)
            lam = self.eigvals[m_abs].to(torch.cdouble)
            column = psi_m[:, phi_idx, :]
            weighted = sqrt_r.unsqueeze(-1).to(torch.cdouble) * column
            radial = (V @ (lam.unsqueeze(-1) * (V.T @ weighted))) / \
                sqrt_r.unsqueeze(-1).to(torch.cdouble)
            spectral = torch.fft.fft(column, dim=-1, norm="forward")
            axial = torch.fft.ifft(
                0.5 * (self.kz.reshape(1, -1) ** 2).to(torch.cdouble) * spectral,
                dim=-1, norm="forward")
            expected[:, phi_idx, :] = radial + axial
        expected = torch.fft.ifft(expected, dim=1, norm="ortho")

        actual = CylindricalGroundState.apply_kinetic(
            psi, self.kz, self.m_modes, self.r, self.eigvecs, self.eigvals)
        self.assertLess(float((expected - actual).abs().max()), 1e-12)

    def test_operator_is_self_adjoint_in_the_weighted_inner_product(self):
        """⟨φ|Tψ⟩ = ⟨Tφ|ψ⟩ under ∫ … r dr dφ dz — the property the √r
        symmetrisation exists to guarantee."""
        psi, phi = self._random(0), self._random(1)
        left = torch.sum(phi.conj() * CylindricalGroundState.apply_kinetic(
            psi, self.kz, self.m_modes, self.r, self.eigvecs, self.eigvals) * self.r_w) * self.dV
        right = torch.sum(CylindricalGroundState.apply_kinetic(
            phi, self.kz, self.m_modes, self.r, self.eigvecs, self.eigvals).conj()
            * psi * self.r_w) * self.dV
        self.assertLess(abs(complex(left - right)), 1e-12)

    def test_expectation_value_is_real_and_non_negative(self):
        """The kinetic energy of a random state is real and positive.

        The operator is self-adjoint and positive semi-definite, so a complex
        expectation value means the sqrt(r) weighting has been applied on only one
        side of the inner product.
        """
        psi = self._random()
        expectation = torch.sum(psi.conj() * CylindricalGroundState.apply_kinetic(
            psi, self.kz, self.m_modes, self.r, self.eigvecs, self.eigvals) * self.r_w) * self.dV
        self.assertLess(abs(float(expectation.imag)), 1e-12)
        self.assertGreater(float(expectation.real), 0.0)

    def test_agrees_with_the_gradient_form_of_the_kinetic_energy(self):
        """⟨ψ|T|ψ⟩ = ½∫|∇ψ|² for a smooth state (integration by parts)."""
        gr, _, gz = self.space_grid
        psi = cyl.normalize(torch.exp(-(gr ** 2 + gz ** 2) / 2).to(torch.cdouble),
                            self.r, self.dr, self.dphi, self.dz)
        operator = float(torch.sum(psi.conj() * CylindricalGroundState.apply_kinetic(
            psi, self.kz, self.m_modes, self.r, self.eigvecs, self.eigvals)
            * self.r_w).real * self.dV)
        gradient = float(cyl.calculate_energy_allocation(
            psi, torch.zeros_like(gr), self.r, self.dr, self.dphi, self.dz,
            self.kz, self.m_modes, u=0.0)['e_kin'].real)
        self.assertAlmostEqual(operator, gradient, delta=5e-3)


class TestCylindricalConvergence(unittest.TestCase):
    """End-to-end behaviour of CylindricalGroundState.find_ground_state."""

    @classmethod
    def setUpClass(cls):
        cls.n_r, cls.n_phi, cls.n_z = 24, 4, 12
        (cls.r, _, _, cls.kz, cls.m_modes,
         cls.dr, cls.dphi, cls.dz, (cls.gr, _, cls.gz)) = cyl.init_grid(
            5.0, -4.0, 4.0, cls.n_r, cls.n_phi, cls.n_z, 'cpu')
        cls.harmonic = 0.5 * (cls.gr ** 2 + cls.gz ** 2)

    def _system(self, potential, **overrides):
        params = {
            "Grid_resolution": [self.n_r, self.n_phi, self.n_z],
            "r_max": 5.0, "z_min": -4.0, "z_max": 4.0,
            "a_ho": 1e-6, "u": 0.0,
        }
        params.update(overrides)
        return _StubSystem(params, potential)

    def _solve(self, system, **kwargs):
        target = os.path.join(tempfile.gettempdir(), "_gs_cyl_test.dat")
        try:
            with contextlib.redirect_stdout(io.StringIO()) as captured:
                psi = CylindricalGroundState.find_ground_state(
                    {}, system, target, 'cpu', **kwargs)
            return psi, captured.getvalue()
        finally:
            if os.path.exists(target):
                os.remove(target)

    def test_result_is_normalised_with_the_cylindrical_volume_element(self):
        """The converged state integrates to 1 under r dr dphi dz.

        Normalising without the r weight leaves a state that looks fine on the grid
        but carries the wrong atom number.
        """
        psi, _ = self._solve(self._system(self.harmonic))
        norm = float(torch.sum(torch.abs(psi) ** 2 * self.r.reshape(-1, 1, 1))
                     * (self.dr * self.dphi * self.dz))
        self.assertAlmostEqual(norm, 1.0, places=10)

    def test_negative_chemical_potential_still_converges(self):
        """A trap shifted down by a constant converges to the same state.

        Subtracting 6 from the potential drives mu negative without changing the
        physics: the density profile must match the unshifted run, since a uniform
        offset only rescales the global phase.
        """
        offset = self.harmonic - 6.0
        psi, _ = self._solve(self._system(offset))
        mu = cyl.calculate_chemical_potential(
            psi, offset, 0.0, self.r, self.dr, self.dphi, self.dz, self.kz, self.m_modes)
        self.assertLess(mu, 0.0)
        reference, _ = self._solve(self._system(self.harmonic))
        self.assertTrue(torch.allclose(torch.abs(psi), torch.abs(reference), atol=1e-3))

    def test_iteration_cap_stops_a_run_that_cannot_converge(self):
        """A run capped at three iterations stops and says it did not converge.

        Strong interactions and a tiny cap guarantee the residual is still large,
        so this pins the cap and its message rather than the convergence itself.
        """
        _, log = self._solve(self._system(self.harmonic, u=200.0), max_iterations=3)
        self.assertIn("did not converge", log)

    def test_state_stays_axisymmetric_for_an_axisymmetric_trap(self):
        """An axisymmetric trap gives a state with no azimuthal structure.

        Every phi slice must match the first to 1e-8: the m != 0 channels are
        driven only by the potential, so any variation here means the azimuthal
        transform has leaked between modes.
        """
        psi, _ = self._solve(self._system(self.harmonic))
        for j in range(1, self.n_phi):
            self.assertLess(float((torch.abs(psi[:, j, :]) - torch.abs(psi[:, 0, :])).abs().max()),
                            1e-8)


if __name__ == '__main__':
    unittest.main()