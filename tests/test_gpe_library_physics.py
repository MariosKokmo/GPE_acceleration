"""Physics-level tests for the Cartesian GPE library.

Where ``test_gpe_library.py`` checks shapes, dtypes and argument handling, this
module pins the *numbers*: energies against closed-form harmonic-oscillator
results, the split-step propagator against norm conservation, and the SGPE
against the reservoir behaviour it is supposed to produce.

Several of these are regressions for defects that produced plausible-looking
but wrong output rather than an exception — a kinetic energy that silently
dropped the imaginary part of ∇ψ, integrals missing their volume element, a
chemical potential that had no effect on the dynamics, and a renormalisation
that cancelled three-body losses.
"""
import sys
import unittest

import numpy as np
import torch

sys.path.append('.')

from src.library.common_utils import CommonUtils as cu
from src.library.gpe_library import GPELibrary as gpe
from src.library.gpe_library import GPE2DLibrary as gpe2d
from src.library.gpe_library import GPE3DLibrary as gpe3d
from src.library.parameters import CONSTANTS


def reference_create_vortices(vortices, x1, x3, n1, n2, n3):
    """Element-by-element vortex phase, written the obvious slow way."""
    phase = torch.zeros((n1, n2, n3), dtype=torch.float64)
    for n in range(vortices.shape[1]):
        vx, vz, q = vortices[0][n], vortices[1][n], vortices[2][n]
        for i in range(n3):
            for k in range(n1):
                if (i != (vz + n3 // 2)) or (k >= (vx + n1 // 2)):
                    y = x3[i] - x3[vz + n3 // 2]
                    t = x1[k] - x1[vx + n1 // 2]
                    x = torch.sqrt(t ** 2 + y ** 2) + t
                    phase[k, :, i] += 2 * q * torch.atan2(y, x)
                else:
                    phase[k, :, i] += q * CONSTANTS.pi
    return phase


class HarmonicFixture(unittest.TestCase):
    """A cubic grid holding the 3-D isotropic harmonic oscillator ground state."""

    @classmethod
    def setUpClass(cls):
        cls.n = 32
        cls.L = 12.0
        cls.x_min = [-cls.L / 2] * 3
        cls.dx = [cls.L / cls.n] * 3
        cls.dp = [2 * np.pi / cls.L] * 3
        cls.d_x = float(np.prod(cls.dx))
        (cls.x1, cls.x2, cls.x3, cls.p1, cls.p2, cls.p3,
         cls.p_sq, cls.space_grid, cls.p_grid) = gpe.init_grid(
            cls.x_min, cls.dx, cls.dp, cls.n, cls.n, cls.n, 'cpu')
        gx, gy, gz = cls.space_grid
        cls.V = 0.5 * (gx ** 2 + gy ** 2 + gz ** 2)
        cls.psi = gpe.normalize(
            torch.exp(-(gx ** 2 + gy ** 2 + gz ** 2) / 2).to(torch.cdouble), cls.d_x)

    def norm_of(self, psi):
        return float(self.d_x * torch.sum(torch.abs(psi) ** 2))


class TestInitGridValues(HarmonicFixture):
    def test_real_axes_start_at_x_min_and_step_by_dx(self):
        self.assertAlmostEqual(float(self.x1[0]), self.x_min[0], places=12)
        self.assertAlmostEqual(float(self.x1[1] - self.x1[0]), self.dx[0], places=12)
        self.assertAlmostEqual(float(self.x1[-1]), self.x_min[0] + (self.n - 1) * self.dx[0],
                               places=12)

    def test_momentum_axes_match_fftfreq(self):
        """The wrapped ordering must line up with torch.fft, or the kinetic
        operator is applied to the wrong modes."""
        expected = 2 * np.pi * torch.fft.fftfreq(self.n, d=self.dx[0], dtype=torch.float64)
        for axis in (self.p1, self.p2, self.p3):
            self.assertTrue(torch.allclose(axis, expected))

    def test_p_sq_is_the_squared_momentum_magnitude(self):
        px, py, pz = self.p_grid
        self.assertTrue(torch.allclose(self.p_sq, px ** 2 + py ** 2 + pz ** 2))
        self.assertEqual(float(self.p_sq[0, 0, 0]), 0.0)

    def test_all_outputs_land_on_the_requested_device(self):
        """Axes used to be built on the host regardless of `device`."""
        tensors = [self.x1, self.x2, self.x3, self.p1, self.p2, self.p3, self.p_sq]
        tensors += list(self.space_grid) + list(self.p_grid)
        for tensor in tensors:
            self.assertEqual(tensor.device.type, 'cpu')
            self.assertIn(tensor.dtype, (torch.float64,))

    def test_meshgrids_use_ij_indexing(self):
        """'xy' indexing would transpose the first two axes against p_sq."""
        gx, gy, _ = self.space_grid
        self.assertTrue(torch.allclose(gx[:, 0, 0], self.x1))
        self.assertTrue(torch.allclose(gy[0, :, 0], self.x2))


class TestEnergyAllocation(HarmonicFixture):
    def test_matches_the_analytic_harmonic_oscillator(self):
        """Virial theorem: e_kin = e_pot = 0.75, E = 1.5 in units of ħω_ho."""
        energies = gpe.calculate_energy_allocation(
            self.psi, self.V, (self.p1, self.p2, self.p3), self.d_x, u=0.0)
        self.assertAlmostEqual(float(energies['e_kin'].real), 0.75, places=5)
        self.assertAlmostEqual(float(energies['e_pot'].real), 0.75, places=5)
        self.assertAlmostEqual(float(energies['E_total'].real), 1.5, places=5)

    def test_energy_is_independent_of_grid_spacing(self):
        """The volume element is what makes this true; without it E scales as 1/d_x."""
        coarse = gpe.calculate_energy_allocation(
            self.psi, self.V, (self.p1, self.p2, self.p3), self.d_x, u=0.0)

        n = 48
        dx = [self.L / n] * 3
        d_x = float(np.prod(dx))
        _, _, _, p1, p2, p3, _, (gx, gy, gz), _ = gpe.init_grid(
            self.x_min, dx, [2 * np.pi / self.L] * 3, n, n, n, 'cpu')
        psi = gpe.normalize(torch.exp(-(gx ** 2 + gy ** 2 + gz ** 2) / 2).to(torch.cdouble), d_x)
        fine = gpe.calculate_energy_allocation(
            psi, 0.5 * (gx ** 2 + gy ** 2 + gz ** 2), (p1, p2, p3), d_x, u=0.0)

        self.assertAlmostEqual(float(coarse['E_total'].real),
                               float(fine['E_total'].real), places=5)

    def test_kinetic_energy_includes_the_phase_gradient(self):
        """Boosting by k adds exactly k²/2; dropping Im(∇ψ) would lose it."""
        gx = self.space_grid[0]
        boosted = self.psi * torch.exp(1j * 1.0 * gx)
        rest = gpe.calculate_energy_allocation(
            self.psi, self.V, (self.p1, self.p2, self.p3), self.d_x, u=0.0)
        moving = gpe.calculate_energy_allocation(
            boosted, self.V, (self.p1, self.p2, self.p3), self.d_x, u=0.0)
        self.assertAlmostEqual(float(moving['e_kin'].real) - float(rest['e_kin'].real),
                               0.5, places=5)

    def test_accepts_both_axis_and_meshgrid_momentum_conventions(self):
        from_axes = gpe.calculate_energy_allocation(
            self.psi, self.V, (self.p1, self.p2, self.p3), self.d_x, u=1.0)
        from_grids = gpe.calculate_energy_allocation(
            self.psi, self.V, self.p_grid, self.d_x, u=1.0)
        self.assertAlmostEqual(float(from_axes['e_kin'].real),
                               float(from_grids['e_kin'].real), places=12)

    def test_interaction_energy_scales_linearly_with_u(self):
        one = gpe.calculate_energy_allocation(
            self.psi, self.V, (self.p1, self.p2, self.p3), self.d_x, u=1.0)
        three = gpe.calculate_energy_allocation(
            self.psi, self.V, (self.p1, self.p2, self.p3), self.d_x, u=3.0)
        self.assertAlmostEqual(3 * float(one['e_int'].real),
                               float(three['e_int'].real), places=12)

    def test_complex_potential_contributes_only_its_real_part(self):
        absorbing = self.V - 2j * torch.ones_like(self.V)
        with_absorber = gpe.calculate_energy_allocation(
            self.psi, absorbing, (self.p1, self.p2, self.p3), self.d_x, u=1.0)
        without = gpe.calculate_energy_allocation(
            self.psi, self.V, (self.p1, self.p2, self.p3), self.d_x, u=1.0)
        self.assertAlmostEqual(float(with_absorber['e_pot'].real),
                               float(without['e_pot'].real), places=12)
        self.assertFalse(torch.is_complex(torch.as_tensor(with_absorber['E_total'])))

    def test_missing_interaction_strength_raises(self):
        with self.assertRaises(ValueError):
            gpe.calculate_energy_allocation(
                self.psi, self.V, (self.p1, self.p2, self.p3), self.d_x)

    def test_total_is_the_sum_of_the_parts(self):
        e = gpe.calculate_energy_allocation(
            self.psi, self.V, (self.p1, self.p2, self.p3), self.d_x, u=2.0)
        self.assertAlmostEqual(
            float(e['E_total'].real),
            float((e['e_kin'] + e['e_pot'] + e['e_int']).real), places=12)


class TestChemicalPotential(HarmonicFixture):
    def test_non_interacting_ground_state(self):
        mu = gpe.calculate_chemical_potential(
            self.psi, self.V, 0.0, (self.p1, self.p2, self.p3), self.d_x)
        self.assertAlmostEqual(mu, 1.5, places=5)

    def test_counts_the_interaction_energy_twice(self):
        e = gpe.calculate_energy_allocation(
            self.psi, self.V, (self.p1, self.p2, self.p3), self.d_x, u=4.0)
        mu = gpe.calculate_chemical_potential(
            self.psi, self.V, 4.0, (self.p1, self.p2, self.p3), self.d_x)
        self.assertAlmostEqual(mu - float(e['E_total'].real),
                               float(e['e_int'].real), places=12)

    def test_returns_a_python_float(self):
        mu = gpe.calculate_chemical_potential(
            self.psi, self.V, 1.0, (self.p1, self.p2, self.p3), self.d_x)
        self.assertIsInstance(mu, float)

    def test_agrees_with_the_ground_state_solvers_own_estimate(self):
        """Two independent implementations of μ must give the same number."""
        from src.library.ground_state import GroundState

        u = 20.0
        psi = self.psi.clone()
        dtau = 0.05 * min(self.dx) ** 2
        for _ in range(1500):
            psi, energy, tol, mu_gs = GroundState.steepest_descent(
                psi, dtau, self.p_sq, self.V, self.d_x, u)
        mu_lib = gpe.calculate_chemical_potential(
            psi, self.V, u, (self.p1, self.p2, self.p3), self.d_x)
        self.assertAlmostEqual(float(mu_gs.real), mu_lib, places=9)


class TestModGradPsiComplex(HarmonicFixture):
    def test_plane_wave_gradient_is_the_wavenumber(self):
        """|∇ e^{ikx}| = |k| — zero if only the real part of ∂ψ is kept."""
        gx = self.space_grid[0]
        k = float(self.p1[2])
        psi = torch.exp(1j * k * gx).to(torch.cdouble)
        grad = gpe.mod_grad_psi(psi, (self.p1, self.p2, self.p3))
        self.assertTrue(torch.allclose(grad, torch.full_like(grad, abs(k)), atol=1e-9))

    def test_matches_the_analytic_gradient_of_a_real_gaussian(self):
        gx, gy, gz = self.space_grid
        psi = torch.exp(-(gx ** 2 + gy ** 2 + gz ** 2) / 2).to(torch.cdouble)
        grad = gpe.mod_grad_psi(psi, (self.p1, self.p2, self.p3))
        expected = torch.sqrt(gx ** 2 + gy ** 2 + gz ** 2) * torch.abs(psi)
        interior = (slice(8, 24), slice(8, 24), slice(8, 24))
        self.assertLess(float((grad[interior] - expected[interior]).abs().max()), 1e-4)

    def test_axes_and_meshgrids_agree(self):
        from_axes = gpe.mod_grad_psi(self.psi, (self.p1, self.p2, self.p3))
        from_grids = gpe.mod_grad_psi(self.psi, self.p_grid)
        self.assertTrue(torch.allclose(from_axes, from_grids, atol=1e-14))

    def test_result_is_non_negative_everywhere(self):
        grad = gpe.mod_grad_psi(self.psi * torch.exp(1j * self.space_grid[0]),
                                (self.p1, self.p2, self.p3))
        self.assertGreaterEqual(float(grad.min()), 0.0)

    def test_rejects_a_four_dimensional_field(self):
        with self.assertRaises(ValueError):
            gpe.mod_grad_psi(torch.ones(2, 2, 2, 2, dtype=torch.cdouble),
                             (self.p1, self.p2, self.p3, self.p1))

    def test_rejects_too_few_momentum_axes(self):
        with self.assertRaises(ValueError):
            gpe.mod_grad_psi(self.psi, (self.p1, self.p2))


class TestSplitStepNorm(HarmonicFixture):
    def test_conserves_norm_for_a_real_potential(self):
        """A unitary propagator needs no renormalisation to hold N fixed."""
        psi = self.psi.clone()
        for _ in range(500):
            psi = gpe.split_step_step(psi, self.V, 0.005, self.p_sq, self.d_x)
        self.assertAlmostEqual(self.norm_of(psi), 1.0, places=10)

    def test_three_body_loss_removes_atoms(self):
        """Regression: an unconditional normalise silently cancelled the loss."""
        psi = self.psi.clone()
        for _ in range(300):
            utot = self.V + 1j * (-50.0 * torch.abs(psi) ** 4)
            psi = gpe.split_step_step(psi, utot, 0.005, self.p_sq, self.d_x)
        self.assertLess(self.norm_of(psi), 0.95)

    def test_renormalise_flag_forces_number_conservation(self):
        utot = self.V + 1j * (-50.0 * torch.abs(self.psi) ** 4)
        out = gpe.split_step_step(self.psi, utot, 0.005, self.p_sq, self.d_x,
                                  renormalise=True)
        self.assertAlmostEqual(self.norm_of(out), 1.0, places=12)

    def test_ground_state_density_is_stationary(self):
        psi = self.psi.clone()
        for _ in range(200):
            psi = gpe.split_step_step(psi, self.V, 0.005, self.p_sq, self.d_x)
        drift = float((torch.abs(psi) ** 2 - torch.abs(self.psi) ** 2).abs().max())
        self.assertLess(drift, 1e-5)

    def test_energy_is_conserved_by_the_propagator(self):
        psi = self.psi * torch.exp(1j * 0.5 * self.space_grid[0])
        before = gpe.calculate_energy_allocation(
            psi, self.V, (self.p1, self.p2, self.p3), self.d_x, u=0.0)['E_total'].real
        for _ in range(200):
            psi = gpe.split_step_step(psi, self.V, 0.005, self.p_sq, self.d_x)
        after = gpe.calculate_energy_allocation(
            psi, self.V, (self.p1, self.p2, self.p3), self.d_x, u=0.0)['E_total'].real
        self.assertAlmostEqual(float(before), float(after), places=4)


class TestSGPEStep(HarmonicFixture):
    def setUp(self):
        self.mu_gs = gpe.calculate_chemical_potential(
            self.psi, self.V, 0.0, (self.p1, self.p2, self.p3), self.d_x)

    def _run(self, mu, steps=400, gamma=0.05, **kwargs):
        psi = self.psi.clone()
        for _ in range(steps):
            psi = gpe.sgpe_step(psi, self.V, mu, gamma, 0.005, self.p_sq, self.d_x, **kwargs)
        return psi

    def test_reservoir_above_mu_grows_the_condensate(self):
        self.assertGreater(self.norm_of(self._run(self.mu_gs + 0.5)), 1.05)

    def test_reservoir_below_mu_shrinks_the_condensate(self):
        self.assertLess(self.norm_of(self._run(self.mu_gs - 0.5)), 0.95)

    def test_ground_state_is_a_fixed_point_at_its_own_mu(self):
        self.assertAlmostEqual(self.norm_of(self._run(self.mu_gs)), 1.0, delta=1e-3)

    def test_growth_is_monotonic_in_mu(self):
        norms = [self.norm_of(self._run(self.mu_gs + delta))
                 for delta in (-0.5, -0.2, 0.2, 0.5)]
        self.assertEqual(norms, sorted(norms))

    def test_mu_has_no_effect_once_the_norm_is_forced(self):
        """The reason renormalise defaults to False: μ enters as a constant
        shift, so rescaling afterwards divides it straight back out."""
        low = self._run(self.mu_gs - 5.0, steps=50, renormalise=True)
        high = self._run(self.mu_gs + 5.0, steps=50, renormalise=True)
        self.assertLess(float((torch.abs(low) ** 2 - torch.abs(high) ** 2).abs().max()), 1e-12)

    def test_zero_damping_reduces_to_unitary_evolution(self):
        psi = self._run(self.mu_gs, steps=100, gamma=0.0)
        self.assertAlmostEqual(self.norm_of(psi), 1.0, places=9)

    def test_damping_lowers_the_energy_of_an_excited_state(self):
        gx, gy, gz = self.space_grid
        excited = gpe.normalize(self.psi * (1.0 + 0.4 * gx ** 2), self.d_x)
        before = gpe.calculate_energy_allocation(
            excited, self.V, (self.p1, self.p2, self.p3), self.d_x, u=0.0)['E_total'].real
        psi = excited
        for _ in range(400):
            psi = gpe.sgpe_step(psi, self.V, self.mu_gs, 0.1, 0.005, self.p_sq, self.d_x)
        psi = gpe.normalize(psi, self.d_x)
        after = gpe.calculate_energy_allocation(
            psi, self.V, (self.p1, self.p2, self.p3), self.d_x, u=0.0)['E_total'].real
        self.assertLess(float(after), float(before))


class TestThermalNoise(HarmonicFixture):
    def test_satisfies_the_fluctuation_dissipation_relation(self):
        gamma, kT, dtau = 0.05, 2.0, 0.01
        torch.manual_seed(11)
        shape = (16, 16, 16)
        samples = torch.stack([
            gpe.generate_thermal_noise(shape, gamma, kT, dtau, self.d_x, 'cpu')
            for _ in range(60)])
        measured = float(torch.mean(torch.abs(samples) ** 2))
        expected = 2.0 * gamma * kT * dtau / self.d_x
        self.assertAlmostEqual(measured / expected, 1.0, delta=0.05)

    def test_amplitude_scales_as_sqrt_of_temperature(self):
        torch.manual_seed(2)
        cold = gpe.generate_thermal_noise((24, 24, 24), 0.05, 1.0, 0.01, self.d_x, 'cpu')
        torch.manual_seed(2)
        hot = gpe.generate_thermal_noise((24, 24, 24), 0.05, 4.0, 0.01, self.d_x, 'cpu')
        self.assertAlmostEqual(float(hot.abs().max() / cold.abs().max()), 2.0, places=10)

    def test_real_and_imaginary_parts_are_independent(self):
        torch.manual_seed(4)
        noise = gpe.generate_thermal_noise((32, 32, 32), 0.05, 1.0, 0.01, self.d_x, 'cpu')
        correlation = float(torch.mean(noise.real * noise.imag)
                            / (noise.real.std() * noise.imag.std()))
        self.assertLess(abs(correlation), 0.02)

    def test_zero_temperature_gives_no_noise(self):
        noise = gpe.generate_thermal_noise((8, 8, 8), 0.05, 0.0, 0.01, self.d_x, 'cpu')
        self.assertEqual(float(noise.abs().max()), 0.0)

    def test_projection_removes_modes_above_the_cutoff(self):
        torch.manual_seed(6)
        shape = (self.n, self.n, self.n)
        projected = gpe.generate_thermal_noise(
            shape, 0.05, 1.0, 0.01, self.d_x, 'cpu', p_sq=self.p_sq, e_cut=2.0)
        spectrum = torch.fft.fftn(projected, norm='forward')
        above_cutoff = spectrum[0.5 * self.p_sq > 2.0]
        self.assertLess(float(above_cutoff.abs().max()), 1e-12)
        self.assertGreater(float(spectrum.abs().max()), 0.0)

    def test_no_projection_without_p_sq(self):
        torch.manual_seed(8)
        plain = gpe.generate_thermal_noise((8, 8, 8), 0.05, 1.0, 0.01, self.d_x, 'cpu')
        torch.manual_seed(8)
        cutoff_only = gpe.generate_thermal_noise((8, 8, 8), 0.05, 1.0, 0.01, self.d_x,
                                                 'cpu', e_cut=1.0)
        self.assertTrue(torch.equal(plain, cutoff_only))


class TestCrossSectionLine(unittest.TestCase):
    """The snapshot buffer is allocated as (shots, n1), so axis=1 must be n1 long."""

    def setUp(self):
        torch.manual_seed(0)
        self.n1, self.n2, self.n3 = 8, 4, 6
        self.psi = torch.randn(self.n1, self.n2, self.n3, dtype=torch.cdouble)

    def test_axis_one_returns_a_profile_along_x(self):
        line = gpe2d.calculate_cross_section_line(self.psi, axis=1)
        self.assertEqual(line.shape, (self.n1,))
        expected = torch.sum(torch.abs(self.psi[:, :, self.n3 // 2]) ** 2, dim=1)
        self.assertTrue(torch.allclose(line, expected))

    def test_axis_two_returns_a_profile_along_z(self):
        line = gpe2d.calculate_cross_section_line(self.psi, axis=2)
        self.assertEqual(line.shape, (self.n3,))
        expected = torch.sum(torch.abs(self.psi[self.n1 // 2, :, :]) ** 2, dim=0)
        self.assertTrue(torch.allclose(line, expected))

    def test_default_axis_is_x(self):
        self.assertTrue(torch.equal(gpe2d.calculate_cross_section_line(self.psi),
                                    gpe2d.calculate_cross_section_line(self.psi, axis=1)))

    def test_fits_the_snapshot_buffer_on_a_non_cubic_grid(self):
        """With n1 != n3 the old orientation raised a shape mismatch here."""
        buffer = torch.zeros(3, self.n1)
        buffer[0, :] = gpe2d.calculate_cross_section_line(self.psi, axis=1)
        self.assertTrue(torch.all(torch.isfinite(buffer[0])))

    def test_values_are_non_negative_densities(self):
        self.assertGreaterEqual(float(gpe2d.calculate_cross_section_line(self.psi).min()), 0.0)

    def test_invalid_axis_raises(self):
        with self.assertRaises(ValueError):
            gpe2d.calculate_cross_section_line(self.psi, axis=3)


class TestCreateVorticesVectorised(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.n1, cls.n2, cls.n3 = 32, 6, 24
        cls.x1, cls.x2, cls.x3, *_ = gpe.init_grid(
            [-6.0, -3.0, -5.0],
            [12 / cls.n1, 6 / cls.n2, 10 / cls.n3],
            [2 * np.pi / 12, 2 * np.pi / 6, 2 * np.pi / 10],
            cls.n1, cls.n2, cls.n3, 'cpu')

    def _create(self, vortices):
        return gpe2d.create_vortices(vortices, self.x1, self.x2, self.x3,
                                     self.n1, self.n2, self.n3, 'cpu')

    def test_matches_the_element_by_element_reference(self):
        vortices = np.array([[0, 7, -5], [0, -6, 3], [1, -2, 3]])
        expected = reference_create_vortices(vortices, self.x1, self.x3,
                                             self.n1, self.n2, self.n3)
        self.assertLess(float((self._create(vortices) - expected).abs().max()), 1e-15)

    def test_single_vortex_winds_by_two_pi_per_unit_charge(self):
        for charge in (1, 2, -1):
            phase = self._create(np.array([[0], [0], [charge]]))
            # Sample a full circle finely enough that no single step reaches pi,
            # otherwise the unwrapping is ambiguous for |charge| >= 2.
            radius, samples = 6, 24
            ring = []
            for s in range(samples):
                angle = 2 * np.pi * s / samples
                k = int(round(radius * np.cos(angle)))
                i = int(round(radius * np.sin(angle)))
                ring.append(float(phase[self.n1 // 2 + k, 0, self.n3 // 2 + i]))
            ring.append(ring[0])
            steps = [ring[j + 1] - ring[j] for j in range(samples)]
            steps = [(s + np.pi) % (2 * np.pi) - np.pi for s in steps]
            self.assertAlmostEqual(sum(steps), 2 * np.pi * charge, places=6)

    def test_phase_is_real_valued(self):
        phase = self._create(np.array([[0], [0], [1]]))
        self.assertFalse(torch.is_complex(phase))
        self.assertEqual(phase.dtype, torch.float64)

    def test_phase_is_uniform_along_y(self):
        phase = self._create(np.array([[3], [2], [1]]))
        for j in range(1, self.n2):
            self.assertTrue(torch.equal(phase[:, j, :], phase[:, 0, :]))

    def test_result_is_finite_everywhere_including_the_core(self):
        phase = self._create(np.array([[0], [0], [1]]))
        self.assertTrue(bool(torch.all(torch.isfinite(phase))))

    def test_opposite_charges_have_zero_net_winding(self):
        """A vortex-antivortex pair is topologically neutral: any loop enclosing
        both accumulates no net phase (the field itself only decays with
        distance, it does not vanish at finite range)."""
        pair = self._create(np.array([[4, -4], [0, 0], [1, -1]]))
        radius, samples = 10, 32
        ring = []
        for s in range(samples):
            angle = 2 * np.pi * s / samples
            k = int(round(radius * np.cos(angle)))
            i = int(round(radius * np.sin(angle)))
            ring.append(float(pair[self.n1 // 2 + k, 0, self.n3 // 2 + i]))
        ring.append(ring[0])
        steps = [(ring[j + 1] - ring[j] + np.pi) % (2 * np.pi) - np.pi
                 for j in range(samples)]
        self.assertAlmostEqual(sum(steps), 0.0, places=6)

    def test_charges_superpose(self):
        """Two co-located unit vortices give the same phase as one of charge 2."""
        two_singles = self._create(np.array([[3, 3], [1, 1], [1, 1]]))
        one_double = self._create(np.array([[3], [1], [2]]))
        self.assertLess(float((two_singles - one_double).abs().max()), 1e-12)

    def test_accepts_a_flat_single_vortex_array(self):
        flat = self._create(np.array([0, 0, 1]))
        shaped = self._create(np.array([[0], [0], [1]]))
        self.assertTrue(torch.equal(flat, shaped))

    def test_none_returns_none(self):
        self.assertIsNone(self._create(None))

    def test_malformed_array_raises(self):
        with self.assertRaises(ValueError):
            self._create(np.array([[0, 1], [0, 1]]))

    def test_out_of_range_vortex_raises(self):
        with self.assertRaises(ValueError):
            self._create(np.array([[self.n1], [0], [1]]))

    def test_imprinting_preserves_density(self):
        """A pure phase imprint must not change |ψ|."""
        from src.library.common_utils import CommonUtils as cu
        psi = torch.ones(self.n1, self.n2, self.n3, dtype=torch.cdouble)
        imprinted = cu.update_phase(psi, self._create(np.array([[0], [0], [1]])))
        self.assertTrue(torch.allclose(torch.abs(imprinted), torch.abs(psi), atol=1e-12))


class TestVelocity3D(HarmonicFixture):
    def test_plane_wave_velocity_is_the_wavenumber(self):
        gx = self.space_grid[0]
        k = float(self.p1[2])
        psi = torch.exp(1j * k * gx).to(torch.cdouble)
        v1, v2, v3 = gpe3d.calculate_velocity3D(psi, self.p_grid)
        self.assertAlmostEqual(float(torch.mean(v1)), k, places=8)
        self.assertAlmostEqual(float(torch.mean(v2)), 0.0, places=8)

    def test_dilute_tail_does_not_produce_huge_velocities(self):
        """`density > 0` is essentially never false in floating point, so the
        cut has to be relative to the peak."""
        gx, gy, gz = self.space_grid
        k = float(self.p1[6])
        psi = (torch.exp(-(gx ** 2 + gy ** 2 + gz ** 2) / 2)
               * torch.exp(1j * k * gx)).to(torch.cdouble)
        components = gpe3d.calculate_velocity3D(psi, self.p_grid)
        for v in components:
            self.assertTrue(bool(torch.all(torch.isfinite(v))))
            # the true velocity is k; nothing anywhere may exceed it by much
            self.assertLess(float(v.abs().max()), 2.0 * abs(k))

    def test_velocity_is_zero_where_the_density_is_negligible(self):
        gx, gy, gz = self.space_grid
        psi = (torch.exp(-(gx ** 2 + gy ** 2 + gz ** 2) / 2)
               * torch.exp(1j * float(self.p1[6]) * gx)).to(torch.cdouble)
        density = torch.abs(psi) ** 2
        empty = density < 1e-12 * density.max()
        self.assertTrue(bool(empty.any()))
        for v in gpe3d.calculate_velocity3D(psi, self.p_grid):
            self.assertEqual(float(v[empty].abs().max()), 0.0)

    def test_density_floor_is_configurable(self):
        gx, gy, gz = self.space_grid
        psi = (torch.exp(-(gx ** 2 + gy ** 2 + gz ** 2) / 2)
               * torch.exp(1j * float(self.p1[6]) * gx)).to(torch.cdouble)
        loose = gpe3d.calculate_velocity3D(psi, self.p_grid, density_floor=1e-1)[0]
        tight = gpe3d.calculate_velocity3D(psi, self.p_grid, density_floor=1e-20)[0]
        self.assertLess(int(torch.count_nonzero(loose)), int(torch.count_nonzero(tight)))

    def test_real_state_has_zero_velocity(self):
        for v in gpe3d.calculate_velocity3D(self.psi, self.p_grid):
            self.assertLess(float(v.abs().max()), 1e-10)


class TestAngularMomentumVolumeElement(HarmonicFixture):
    def _vortex_state(self, charge=1):
        gx, gy, gz = self.space_grid
        rho = torch.sqrt(gx ** 2 + gy ** 2)
        psi = (rho ** abs(charge)
               * torch.exp(-(gx ** 2 + gy ** 2 + gz ** 2) / 2)
               * torch.exp(1j * charge * torch.atan2(gy, gx)))
        return gpe.normalize(psi.to(torch.cdouble), self.d_x)

    def test_charge_one_vortex_carries_one_hbar(self):
        lz = gpe3d.angular_momentum(self._vortex_state(1), self.space_grid,
                                    self.p_grid, 3, self.d_x)
        self.assertAlmostEqual(float(lz), 1.0, places=3)

    def test_charge_two_vortex_carries_two_hbar(self):
        lz = gpe3d.angular_momentum(self._vortex_state(2), self.space_grid,
                                    self.p_grid, 3, self.d_x)
        self.assertAlmostEqual(float(lz), 2.0, places=3)

    def test_negative_charge_reverses_the_sign(self):
        lz = gpe3d.angular_momentum(self._vortex_state(-1), self.space_grid,
                                    self.p_grid, 3, self.d_x)
        self.assertAlmostEqual(float(lz), -1.0, places=3)

    def test_symmetric_real_state_carries_none(self):
        for component in (1, 2, 3):
            lz = gpe3d.angular_momentum(self.psi, self.space_grid, self.p_grid,
                                        component, self.d_x)
            self.assertAlmostEqual(float(lz), 0.0, places=8)

    def test_scales_with_the_volume_element(self):
        """Omitting d_x inflates the expectation value by 1/d_x."""
        state = self._vortex_state(1)
        with_dx = gpe3d.angular_momentum(state, self.space_grid, self.p_grid, 3, self.d_x)
        unit = gpe3d.angular_momentum(state, self.space_grid, self.p_grid, 3, 1.0)
        self.assertAlmostEqual(float(with_dx), float(unit) * self.d_x, places=12)

    def test_invalid_component_raises(self):
        with self.assertRaises(ValueError):
            gpe3d.angular_momentum(self.psi, self.space_grid, self.p_grid, 4, self.d_x)


class TestColumnDensitySpacing(unittest.TestCase):
    def test_default_is_a_bare_sum(self):
        psi = torch.ones(4, 5, 6, dtype=torch.cdouble)
        self.assertAlmostEqual(float(gpe3d.column_density(psi, 1)[0, 0]), 4.0, places=12)

    def test_spacing_turns_it_into_a_line_integral(self):
        psi = torch.ones(4, 5, 6, dtype=torch.cdouble)
        column = gpe3d.column_density(psi, 1, d_axis=0.25)
        self.assertAlmostEqual(float(column[0, 0]), 1.0, places=12)

    def test_each_axis_reduces_the_right_dimension(self):
        psi = torch.ones(4, 5, 6, dtype=torch.cdouble)
        self.assertEqual(gpe3d.column_density(psi, 1).shape, (5, 6))
        self.assertEqual(gpe3d.column_density(psi, 2).shape, (4, 6))
        self.assertEqual(gpe3d.column_density(psi, 3).shape, (4, 5))


class TestVelocity2DPlane(HarmonicFixture):
    """calculate_velocity2D takes the wavefunction, not its phase."""

    def test_plane_wave_along_x(self):
        gx = self.space_grid[0]
        k = float(self.p1[3])
        mod, angle = gpe2d.calculate_velocity2D(
            torch.exp(1j * k * gx).to(torch.cdouble), self.p_grid)
        self.assertTrue(torch.allclose(mod, torch.full_like(mod, abs(k)), atol=1e-9))
        # flow is along +x, so the in-plane direction is zero
        self.assertLess(float(torch.sin(angle).abs().max()), 1e-9)

    def test_plane_wave_along_z_is_the_second_in_plane_component(self):
        """The model plane is x–z, so a z-dependent phase must register and a
        y-dependent one must not."""
        _, gy, gz = self.space_grid
        k = float(self.p3[3])
        mod_z, angle_z = gpe2d.calculate_velocity2D(
            torch.exp(1j * k * gz).to(torch.cdouble), self.p_grid)
        mod_y, _ = gpe2d.calculate_velocity2D(
            torch.exp(1j * k * gy).to(torch.cdouble), self.p_grid)
        self.assertTrue(torch.allclose(mod_z, torch.full_like(mod_z, abs(k)), atol=1e-9))
        self.assertLess(float(mod_y.abs().max()), 1e-9)
        # flow along +z is a quarter turn from +x
        self.assertAlmostEqual(float(angle_z.mean()), np.pi / 2, places=6)

    def test_agrees_with_the_three_dimensional_routine(self):
        gx, _, gz = self.space_grid
        psi = (self.psi * torch.exp(1j * (0.7 * gx + 0.3 * gz))).to(torch.cdouble)
        mod, _ = gpe2d.calculate_velocity2D(psi, self.p_grid)
        v1, _, v3 = gpe3d.calculate_velocity3D(psi, self.p_grid)
        self.assertTrue(torch.allclose(mod, torch.sqrt(v1 ** 2 + v3 ** 2), atol=1e-14))

    def test_real_state_is_at_rest(self):
        mod, _ = gpe2d.calculate_velocity2D(self.psi, self.p_grid)
        self.assertLess(float(mod.abs().max()), 1e-10)


class VortexPairFixture(unittest.TestCase):
    """A vortex-antivortex pair, whose superfluid velocity is known exactly.

    For psi = f(r1) f(r2) exp(i(phi1 - phi2)) with a real envelope f, the
    velocity is exactly grad(phi1 - phi2), i.e.

        v_x = -z/r1^2 + z/r2^2 ,   v_z = (x-d)/r1^2 - (x+d)/r2^2

    The pair carries zero net circulation, so the state is compatible with the
    periodic grid; the remaining far-field mismatch across the boundary is what
    sets the residual error, and it shrinks as the box grows.
    """

    @classmethod
    def setUpClass(cls):
        cls.n, cls.L, cls.xi, cls.d = 256, 40.0, 0.8, 4.0
        dx = [cls.L / cls.n, cls.L / 4, cls.L / cls.n]
        dp = [2 * np.pi / cls.L, 2 * np.pi / (cls.L / 4), 2 * np.pi / cls.L]
        (_, _, _, cls.p1, _, _, _, (gx, gy, gz), cls.p_grid) = gpe.init_grid(
            [-cls.L / 2, -cls.L / 8, -cls.L / 2], dx, dp, cls.n, 4, cls.n, 'cpu')
        cls.gx, cls.gz = gx, gz
        cls.r1 = torch.sqrt((gx - cls.d) ** 2 + gz ** 2)
        cls.r2 = torch.sqrt((gx + cls.d) ** 2 + gz ** 2)
        theta = torch.atan2(gz, gx - cls.d) - torch.atan2(gz, gx + cls.d)
        cls.psi = (torch.tanh(cls.r1 / cls.xi) * torch.tanh(cls.r2 / cls.xi)
                   * torch.exp(1j * theta)).to(torch.cdouble)
        cls.vx = -gz / cls.r1 ** 2 + gz / cls.r2 ** 2
        cls.vz = (gx - cls.d) / cls.r1 ** 2 - (gx + cls.d) / cls.r2 ** 2
        cls.mod_true = torch.sqrt(cls.vx ** 2 + cls.vz ** 2)
        cls.slice_y = 2
        cls.mask = ((cls.r1[:, 2, :] > 3 * cls.xi) & (cls.r2[:, 2, :] > 3 * cls.xi)
                    & (gx[:, 2, :].abs() < cls.L / 4) & (gz[:, 2, :].abs() < cls.L / 4))

    def relative_error(self, mod):
        return ((mod[:, self.slice_y, :] - self.mod_true[:, self.slice_y, :]).abs()
                / self.mod_true[:, self.slice_y, :])[self.mask]


class TestVelocityAgainstAnalyticVortexPair(VortexPairFixture):
    def test_magnitude_matches_the_analytic_field(self):
        """A few percent; the residual is the fixture's own boundary mismatch,
        not the method — see test_error_shrinks_as_the_box_grows."""
        mod, _ = gpe2d.calculate_velocity2D(self.psi, self.p_grid)
        error = self.relative_error(mod)
        self.assertLess(float(error.median()), 0.05)
        self.assertLess(float(error.quantile(0.9)), 0.20)

    def test_error_shrinks_as_the_box_grows(self):
        """The dipole's far field decays as 1/rho^2, so it matches across the
        periodic boundary ever better on a larger box while the grid spacing
        stays fixed. A method error would not behave this way — it pins the
        residual on the fixture rather than on calculate_velocity2D."""
        errors = []
        for L, n in ((20.0, 128), (40.0, 256)):
            dx = [L / n, L / 4, L / n]
            dp = [2 * np.pi / L, 2 * np.pi / (L / 4), 2 * np.pi / L]
            _, _, _, _, _, _, _, (gx, _, gz), p_grid = gpe.init_grid(
                [-L / 2, -L / 8, -L / 2], dx, dp, n, 4, n, 'cpu')
            r1 = torch.sqrt((gx - self.d) ** 2 + gz ** 2)
            r2 = torch.sqrt((gx + self.d) ** 2 + gz ** 2)
            theta = torch.atan2(gz, gx - self.d) - torch.atan2(gz, gx + self.d)
            psi = (torch.tanh(r1 / self.xi) * torch.tanh(r2 / self.xi)
                   * torch.exp(1j * theta)).to(torch.cdouble)
            true = torch.sqrt((-gz / r1 ** 2 + gz / r2 ** 2) ** 2
                              + ((gx - self.d) / r1 ** 2 - (gx + self.d) / r2 ** 2) ** 2)
            mod, _ = gpe2d.calculate_velocity2D(psi, p_grid)
            mask = ((r1[:, 2, :] > 3 * self.xi) & (r2[:, 2, :] > 3 * self.xi)
                    & (gx[:, 2, :].abs() < 6) & (gz[:, 2, :].abs() < 6))
            errors.append(float((((mod[:, 2, :] - true[:, 2, :]).abs()
                                  / true[:, 2, :])[mask]).median()))
        self.assertLess(errors[1], errors[0] / 3.0,
                        msg=f"error did not converge with box size: {errors}")

    def test_direction_matches_the_analytic_field(self):
        _, angle = gpe2d.calculate_velocity2D(self.psi, self.p_grid)
        expected = torch.atan2(self.vz, self.vx)
        delta = (angle - expected + np.pi) % (2 * np.pi) - np.pi
        self.assertLess(float(delta[:, self.slice_y, :][self.mask].abs().median()), 0.05)

    def test_circulation_around_one_core_is_two_pi(self):
        """The line integral of v around a single core must be quantised."""
        v1, _, v3 = gpe3d.calculate_velocity3D(self.psi, self.p_grid)
        centre = self.n // 2 + int(round(self.d / (self.L / self.n)))
        radius, samples = 24, 200
        total = 0.0
        for s in range(samples):
            a = 2 * np.pi * s / samples
            i = centre + int(round(radius * np.cos(a)))
            k = self.n // 2 + int(round(radius * np.sin(a)))
            # v . dl  with dl = radius * (-sin a, cos a) * dtheta, in grid units
            step = 2 * np.pi / samples * radius * (self.L / self.n)
            total += float(v1[i, self.slice_y, k]) * (-np.sin(a)) * step
            total += float(v3[i, self.slice_y, k]) * (np.cos(a)) * step
        self.assertAlmostEqual(total, 2 * np.pi, delta=0.3)

    def test_differentiating_the_wrapped_phase_is_far_worse(self):
        """Regression guard for the branch-cut defect.

        Taking the gradient of angle(psi) instead of using Im(psi* grad psi)
        rings globally, because a spectral derivative of the 2*pi jump is not a
        local operation. How bad it gets depends on how much of the box the cut
        spans — on a box 2.5x the vortex separation (the realistic case, built
        here) it is over 100%, on a 10x box around 10% — so the guard is the
        *ratio* between the two routes rather than an absolute figure.
        """
        L, n = 20.0, 128
        dx = [L / n, L / 4, L / n]
        dp = [2 * np.pi / L, 2 * np.pi / (L / 4), 2 * np.pi / L]
        _, _, _, _, _, _, _, (gx, _, gz), p_grid = gpe.init_grid(
            [-L / 2, -L / 8, -L / 2], dx, dp, n, 4, n, 'cpu')
        r1 = torch.sqrt((gx - self.d) ** 2 + gz ** 2)
        r2 = torch.sqrt((gx + self.d) ** 2 + gz ** 2)
        theta = torch.atan2(gz, gx - self.d) - torch.atan2(gz, gx + self.d)
        psi = (torch.tanh(r1 / self.xi) * torch.tanh(r2 / self.xi)
               * torch.exp(1j * theta)).to(torch.cdouble)
        true = torch.sqrt((-gz / r1 ** 2 + gz / r2 ** 2) ** 2
                          + ((gx - self.d) / r1 ** 2 - (gx + self.d) / r2 ** 2) ** 2)
        mask = ((r1[:, 2, :] > 3 * self.xi) & (r2[:, 2, :] > 3 * self.xi)
                & (gx[:, 2, :].abs() < 6) & (gz[:, 2, :].abs() < 6))

        def median_error(mod):
            return float((((mod[:, 2, :] - true[:, 2, :]).abs()
                           / true[:, 2, :])[mask]).median())

        from_psi, _ = gpe2d.calculate_velocity2D(psi, p_grid)

        axes = gpe._broadcast_momentum_axes(p_grid, 3, zero_nyquist=True)
        phase_f = torch.fft.fftn(torch.angle(psi))
        gradx = torch.fft.ifftn(1j * axes[0] * phase_f).real
        gradz = torch.fft.ifftn(1j * axes[2] * phase_f).real
        from_phase = torch.sqrt(gradx ** 2 + gradz ** 2)

        psi_error, phase_error = median_error(from_psi), median_error(from_phase)
        self.assertLess(psi_error, 0.10)
        self.assertGreater(phase_error, 1.0)
        self.assertGreater(phase_error, 10 * psi_error,
                           msg=f"psi route {psi_error:.3%}, phase route {phase_error:.3%}")

    def test_no_nan_when_a_core_sits_exactly_on_a_grid_point(self):
        """psi = 0 there; the old phase extraction returned NaN, and a single
        NaN spreads over the entire array through the FFT."""
        rho = torch.sqrt(self.gx ** 2 + self.gz ** 2)
        centred = (torch.tanh(rho / self.xi)
                   * torch.exp(1j * torch.atan2(self.gz, self.gx))).to(torch.cdouble)
        self.assertTrue(bool((torch.abs(centred) == 0).any()),
                        msg="fixture should place a node exactly on a grid point")
        mod, angle = gpe2d.calculate_velocity2D(centred, self.p_grid)
        self.assertEqual(int(torch.isnan(mod).sum()), 0)
        self.assertEqual(int(torch.isnan(angle).sum()), 0)


class TestExtractPhase(unittest.TestCase):
    def test_agrees_with_the_logarithmic_form_away_from_nodes(self):
        torch.manual_seed(0)
        psi = torch.randn(6, 5, 4, dtype=torch.cdouble)
        legacy = torch.imag(torch.log(psi / torch.sqrt(torch.abs(psi) ** 2)))
        self.assertTrue(torch.allclose(cu.extract_phase(psi), legacy, atol=1e-12))

    def test_is_finite_at_a_node(self):
        psi = torch.tensor([1 + 1j, 0 + 0j, -2 + 0j], dtype=torch.cdouble)
        phase = cu.extract_phase(psi)
        self.assertEqual(int(torch.isnan(phase).sum()), 0)
        self.assertEqual(float(phase[1]), 0.0)

    def test_is_wrapped_to_the_principal_branch(self):
        torch.manual_seed(1)
        phase = cu.extract_phase(torch.randn(500, dtype=torch.cdouble))
        self.assertLessEqual(float(phase.max()), np.pi)
        self.assertGreater(float(phase.min()), -np.pi - 1e-12)

    def test_recovers_an_imprinted_phase(self):
        gx = torch.linspace(-1.0, 1.0, 16, dtype=torch.float64)
        imprinted = cu.update_phase(torch.ones(16, dtype=torch.cdouble), gx)
        self.assertTrue(torch.allclose(cu.extract_phase(imprinted), gx, atol=1e-12))

    def test_output_is_real(self):
        psi = torch.randn(4, 4, dtype=torch.cdouble)
        self.assertFalse(torch.is_complex(cu.extract_phase(psi)))


class TestDarkSolitonValidation(unittest.TestCase):
    def setUp(self):
        self.n1, self.n2, self.n3 = 16, 4, 16
        self.x1 = torch.linspace(-4, 4, self.n1, dtype=torch.float64)
        self.x3 = torch.linspace(-4, 4, self.n3, dtype=torch.float64)

    def _make(self, **kwargs):
        return gpe2d.create_dark_soliton(self.x1, self.x3, self.n1, self.n2, self.n3,
                                         device=torch.device('cpu'), **kwargs)

    def test_mismatched_widths_raise(self):
        with self.assertRaises(ValueError):
            self._make(positions=[0.0, 1.0], widths=[1.0], axes=[1, 1])

    def test_mismatched_axes_raise(self):
        with self.assertRaises(ValueError):
            self._make(positions=[0.0, 1.0], widths=[1.0, 1.0], axes=[1])

    def test_mismatched_greyness_raises(self):
        with self.assertRaises(ValueError):
            self._make(positions=[0.0, 1.0], widths=[1.0, 1.0], axes=[1, 1],
                       greyness=[0.0])

    def test_matched_lengths_are_accepted(self):
        mask = self._make(positions=[0.0, 1.0], widths=[1.0, 1.0], axes=[1, 3],
                          greyness=[0.0, 0.3])
        self.assertEqual(mask.shape, (self.n1, self.n2, self.n3))


if __name__ == '__main__':
    unittest.main()
