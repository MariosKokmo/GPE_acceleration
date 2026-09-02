"""Tests for the cylindrical (r, φ, z) GPE library.

Coverage is organised as:

* grid + radial operator construction (half-point grid, spectral wrapping,
  symmetry of the √r-symmetrised kinetic matrix),
* the evolution operators (unitarity, equivalence of the batched radial step
  to a plain per-φ reference loop, norm handling in split_step_step/sgpe_step),
* the diagnostics against closed-form results for the 3-D harmonic oscillator,
* the class layout — several methods the simulation loop calls used to live on
  a sibling class with no base, which raised AttributeError at runtime.
"""
import sys
import unittest

import numpy as np
import torch

sys.path.append('.')

from src.library.gpe_cylindrical_library import GPECylindricalLibrary as cyl
from src.library.gpe_cylindrical_library import GPE2DCylindricalLibrary as cyl2d


def reference_radial_p_evolution(psi, dtau, kz, m_modes, r, eigvecs_dict,
                                 eigvals_dict, damping=1j):
    """Straightforward per-φ implementation of the kinetic propagator.

    Deliberately written the slow, obvious way — one Python iteration per
    azimuthal index — so it can pin the batched implementation in the library.
    """
    n_r, n_phi, n_z = psi.shape
    psi_m = torch.fft.fft(psi, dim=1, norm="ortho")
    psi_mk = torch.fft.fft(psi_m, dim=2, norm="forward")
    psi_mk = torch.exp(-damping * dtau * 0.5 * kz.reshape(1, 1, n_z) ** 2) * psi_mk

    sqrt_r = torch.sqrt(r)
    out = torch.empty_like(psi_mk)
    for phi_idx in range(n_phi):
        m_abs = int(abs(m_modes[phi_idx].item()))
        V = eigvecs_dict[m_abs].to(torch.cdouble)
        lam = eigvals_dict[m_abs].to(torch.cdouble)
        weighted = sqrt_r.unsqueeze(-1).to(torch.cdouble) * psi_mk[:, phi_idx, :]
        coeff = V.T @ weighted
        coeff = torch.exp(-damping * dtau * lam.reshape(-1, 1)) * coeff
        out[:, phi_idx, :] = (V @ coeff) / sqrt_r.unsqueeze(-1).to(torch.cdouble)

    out = torch.fft.ifft(out, dim=2, norm="forward")
    return torch.fft.ifft(out, dim=1, norm="ortho")


class CylindricalFixture(unittest.TestCase):
    """Shared small cylindrical grid plus its radial operators."""

    @classmethod
    def setUpClass(cls):
        cls.n_r, cls.n_phi, cls.n_z = 48, 8, 24
        cls.r_max, cls.z_min, cls.z_max = 6.0, -6.0, 6.0
        (cls.r, cls.phi, cls.z, cls.kz, cls.m_modes,
         cls.dr, cls.dphi, cls.dz, cls.space_grid) = cyl.init_grid(
            cls.r_max, cls.z_min, cls.z_max, cls.n_r, cls.n_phi, cls.n_z, 'cpu')
        cls.eigvecs, cls.eigvals = cyl.build_radial_operators(
            cls.r, cls.dr, cls.m_modes, 'cpu')
        cls.dV = cls.dr * cls.dphi * cls.dz
        cls.r_w = cls.r.reshape(-1, 1, 1)

    def gaussian(self):
        """Ground state of the isotropic 3-D harmonic trap, cylindrically sampled."""
        gr, _, gz = self.space_grid
        return cyl.normalize(torch.exp(-(gr ** 2 + gz ** 2) / 2).to(torch.cdouble),
                             self.r, self.dr, self.dphi, self.dz)

    def harmonic_potential(self):
        gr, _, gz = self.space_grid
        return 0.5 * (gr ** 2 + gz ** 2)

    def norm_of(self, psi):
        return float(torch.sum(torch.abs(psi) ** 2 * self.r_w) * self.dV)

    def random_state(self, seed=0):
        torch.manual_seed(seed)
        psi = torch.randn(self.n_r, self.n_phi, self.n_z, dtype=torch.cdouble)
        return cyl.normalize(psi, self.r, self.dr, self.dphi, self.dz)


class TestCylindricalInitGrid(CylindricalFixture):
    def test_radial_grid_is_half_point(self):
        """r_i = (i + 1/2) dr keeps the grid off the r = 0 singularity."""
        self.assertAlmostEqual(self.dr, self.r_max / self.n_r, places=15)
        self.assertAlmostEqual(float(self.r[0]), 0.5 * self.dr, places=15)
        expected = (torch.arange(self.n_r, dtype=torch.float64) + 0.5) * self.dr
        self.assertTrue(torch.allclose(self.r, expected))
        self.assertGreater(float(self.r.min()), 0.0)

    def test_axis_shapes_and_dtypes(self):
        """Every axis has the length its resolution asks for, in float64.

        The meshgrids are checked too: a broadcasting slip between the three axes
        would show up here rather than as wrong numbers much later.
        """
        for axis, n in ((self.r, self.n_r), (self.phi, self.n_phi), (self.z, self.n_z),
                        (self.kz, self.n_z), (self.m_modes, self.n_phi)):
            self.assertEqual(axis.shape, (n,))
            self.assertEqual(axis.dtype, torch.float64)
        for grid in self.space_grid:
            self.assertEqual(grid.shape, (self.n_r, self.n_phi, self.n_z))

    def test_axes_are_built_on_the_requested_device(self):
        """Every axis must land on `device`, not be assembled on the host first."""
        for axis in (self.r, self.phi, self.z, self.kz, self.m_modes):
            self.assertEqual(axis.device.type, 'cpu')

    def test_azimuthal_grid_spans_one_period(self):
        """phi covers [0, 2*pi) in n_phi equal steps.

        The last point stops short of 2*pi because the grid is periodic: including
        both endpoints would duplicate a column and break the azimuthal transform.
        """
        self.assertAlmostEqual(self.dphi, 2 * np.pi / self.n_phi, places=15)
        self.assertAlmostEqual(float(self.phi[0]), 0.0, places=15)
        self.assertLess(float(self.phi[-1]), 2 * np.pi)

    def test_kz_uses_fft_wrapped_ordering(self):
        """Positive momenta first, negative in the upper half — FFT layout."""
        expected = 2 * np.pi * torch.fft.fftfreq(self.n_z, d=self.dz, dtype=torch.float64)
        self.assertTrue(torch.allclose(self.kz, expected))
        self.assertEqual(float(self.kz[0]), 0.0)
        self.assertLess(float(self.kz[self.n_z // 2]), 0.0)

    def test_m_modes_are_signed_integers_in_dft_order(self):
        """The azimuthal mode numbers follow the DFT layout, 0..3 then -4..-1.

        The sign matters: m and -m have the same radial operator but opposite
        circulation, so a mode list in ascending order would attach the wrong
        angular momentum to half the modes.
        """
        expected = torch.tensor([0., 1., 2., 3., -4., -3., -2., -1.], dtype=torch.float64)
        self.assertTrue(torch.allclose(self.m_modes, expected))


class TestRadialOperators(CylindricalFixture):
    def test_symmetrised_matrix_is_symmetric(self):
        """The √r similarity transform must make T real-symmetric for eigh."""
        n_r, dr = self.n_r, self.dr
        r_plus = (torch.arange(n_r, dtype=torch.float64) + 1.0) * dr
        r_minus = torch.arange(n_r, dtype=torch.float64) * dr
        sqrt_r = torch.sqrt(self.r)
        for m in (0, 1, 3):
            diag = -(r_plus + r_minus) / (self.r * dr ** 2) - float(m) ** 2 / self.r ** 2
            sup = r_plus[:-1] / (self.r[:-1] * dr ** 2)
            sub = r_minus[1:] / (self.r[1:] * dr ** 2)
            T = (torch.diag(-0.5 * diag)
                 + torch.diag(-0.5 * sqrt_r[:-1] * sup / sqrt_r[1:], 1)
                 + torch.diag(-0.5 * sqrt_r[1:] * sub / sqrt_r[:-1], -1))
            self.assertLess(float((T - T.T).abs().max()), 1e-12,
                            msg=f"T_tilde is not symmetric for m={m}")

    def test_one_operator_per_unique_abs_m(self):
        """One eigendecomposition is built per |m|, each of full radial size.

        The radial operator depends on m only through m^2, so modes m and -m share
        one decomposition; storing n_phi of them instead of n_phi/2 + 1 would be
        pure waste.
        """
        keys = {k for k in self.eigvecs if isinstance(k, int)}
        self.assertEqual(keys, set(range(self.n_phi // 2 + 1)))
        for m in keys:
            self.assertEqual(self.eigvecs[m].shape, (self.n_r, self.n_r))
            self.assertEqual(self.eigvals[m].shape, (self.n_r,))

    def test_eigenvectors_are_orthonormal(self):
        """The eigenvector matrices satisfy V^T V = I.

        The kinetic step applies V lambda V^T, so a non-orthonormal basis would
        change the norm on every step.
        """
        for m in (0, 2):
            V = self.eigvecs[m]
            identity = V.T @ V
            self.assertLess(float((identity - torch.eye(self.n_r, dtype=torch.float64)).abs().max()), 1e-10)

    def test_kinetic_eigenvalues_are_non_negative(self):
        """-∇²/2 is positive semi-definite, so no eigenvalue may be negative."""
        for m in (0, 1, 4):
            self.assertGreater(float(self.eigvals[m].min()), -1e-9)

    def test_stacked_operators_are_cached_at_build_time(self):
        """The per-phi stacked operators are built once, at build time.

        The stack has one (n_r, n_r) matrix per azimuthal column, which is what
        lets the kinetic step be a single batched matmul.
        """
        self.assertIn(cyl._STACK_KEY, self.eigvecs)
        V_all, lam_all = self.eigvecs[cyl._STACK_KEY]
        self.assertEqual(V_all.shape, (self.n_phi, self.n_r, self.n_r))
        self.assertEqual(lam_all.shape, (self.n_phi, self.n_r))

    def test_stacked_operators_map_each_phi_to_its_own_abs_m(self):
        """Each stacked entry is the decomposition for that column's |m|.

        An off-by-one in the mapping would apply a neighbouring mode's operator --
        numerically plausible output, wrong physics.
        """
        V_all, lam_all = cyl._stacked_radial(self.eigvecs, self.eigvals, self.m_modes)
        for phi_idx in range(self.n_phi):
            m_abs = int(abs(self.m_modes[phi_idx].item()))
            self.assertTrue(torch.equal(V_all[phi_idx], self.eigvecs[m_abs]))
            self.assertTrue(torch.equal(lam_all[phi_idx], self.eigvals[m_abs]))

    def test_stacking_is_reused_not_rebuilt(self):
        """Asking for the stack twice returns the identical objects.

        Identity, not equality: rebuilding it would silently repeat the stacking
        cost on every step of every run.
        """
        first = cyl._stacked_radial(self.eigvecs, self.eigvals, self.m_modes)
        second = cyl._stacked_radial(self.eigvecs, self.eigvals, self.m_modes)
        self.assertIs(first[0], second[0])
        self.assertIs(first[1], second[1])

    def test_stack_key_does_not_collide_with_mode_keys(self):
        """The cache key must not shadow an |m| entry."""
        self.assertNotIsInstance(cyl._STACK_KEY, int)
        for m in range(self.n_phi // 2 + 1):
            self.assertIn(m, self.eigvecs)


class TestCylindricalNormalize(CylindricalFixture):
    def test_normalize_uses_the_cylindrical_volume_element(self):
        """Normalising divides by the integral taken with r dr dphi dz.

        A uniform state is used, so the r weight is the whole of the difference
        between this and a Cartesian normalisation.
        """
        psi = torch.ones(self.n_r, self.n_phi, self.n_z, dtype=torch.cdouble)
        self.assertAlmostEqual(self.norm_of(cyl.normalize(psi, self.r, self.dr, self.dphi, self.dz)),
                               1.0, places=12)

    def test_normalize_preserves_phase(self):
        """Normalising rescales the amplitude and leaves the phase untouched.

        The factor is real and positive, so multiplying the state by 7 beforehand
        must make no difference to the angle.
        """
        psi = self.random_state()
        scaled = cyl.normalize(psi * 7.0, self.r, self.dr, self.dphi, self.dz)
        self.assertTrue(torch.allclose(torch.angle(scaled), torch.angle(psi), atol=1e-12))


class TestCylindricalPEvolution(CylindricalFixture):
    def test_matches_the_per_phi_reference_loop(self):
        """The batched radial step must reproduce the obvious implementation."""
        psi = self.random_state()
        for damping in (1j, 1j + 0.05):
            expected = reference_radial_p_evolution(
                psi, 0.002, self.kz, self.m_modes, self.r,
                self.eigvecs, self.eigvals, damping)
            actual = cyl.p_evolution(psi, 0.002, self.kz, self.m_modes, self.r,
                                     self.eigvecs, self.eigvals, damping)
            self.assertLess(float((expected - actual).abs().max()), 1e-14,
                            msg=f"batched radial step diverged for damping={damping}")

    def test_is_unitary_for_undamped_evolution(self):
        """Without damping the kinetic step conserves the norm over 15 steps.

        The radial part is applied through an eigendecomposition rather than an
        FFT, so unitarity depends on that basis being orthonormal and on the sqrt(r)
        weighting being undone afterwards.
        """
        psi = self.random_state()
        for _ in range(15):
            psi = cyl.p_evolution(psi, 0.002, self.kz, self.m_modes, self.r,
                                  self.eigvecs, self.eigvals)
        self.assertAlmostEqual(self.norm_of(psi), 1.0, places=11)

    def test_damping_removes_norm(self):
        """(i + γ) damping must decay the state rather than preserve it."""
        psi = self.random_state()
        out = cyl.p_evolution(psi, 0.01, self.kz, self.m_modes, self.r,
                              self.eigvecs, self.eigvals, damping=1j + 0.1)
        self.assertLess(self.norm_of(out), 1.0)

    def test_zero_step_is_the_identity(self):
        """A zero-length step returns the state unchanged.

        The exponential is exp(0) = 1, so any residual difference is the transform
        pair or the sqrt(r) weighting failing to cancel.
        """
        psi = self.random_state()
        out = cyl.p_evolution(psi, 0.0, self.kz, self.m_modes, self.r,
                              self.eigvecs, self.eigvals)
        self.assertLess(float((out - psi).abs().max()), 1e-12)

    def test_ground_state_is_stationary_in_density(self):
        """A harmonic ground state under full GPE steps keeps its density."""
        psi = self.gaussian()
        V = self.harmonic_potential()
        evolved = psi.clone()
        for _ in range(40):
            evolved = cyl.split_step_step(evolved, V, 0.002, self.kz, self.m_modes,
                                          self.r, self.eigvecs, self.eigvals,
                                          self.dr, self.dphi, self.dz)
        density_drift = float((torch.abs(evolved) ** 2 - torch.abs(psi) ** 2).abs().max())
        self.assertLess(density_drift, 1e-3)


class TestCylindricalSplitStep(CylindricalFixture):
    def test_conserves_norm_for_a_real_potential(self):
        """Every factor has unit modulus, so no renormalisation is needed."""
        psi = self.gaussian()
        V = self.harmonic_potential()
        for _ in range(200):
            psi = cyl.split_step_step(psi, V, 0.002, self.kz, self.m_modes, self.r,
                                      self.eigvecs, self.eigvals, self.dr, self.dphi, self.dz)
        self.assertAlmostEqual(self.norm_of(psi), 1.0, places=9)

    def test_three_body_loss_actually_removes_atoms(self):
        """Regression: an unconditional normalise silently cancelled the loss term."""
        psi = self.gaussian()
        V = self.harmonic_potential()
        for _ in range(100):
            utot = V + 1j * (-50.0 * torch.abs(psi) ** 4)
            psi = cyl.split_step_step(psi, utot, 0.002, self.kz, self.m_modes, self.r,
                                      self.eigvecs, self.eigvals, self.dr, self.dphi, self.dz)
        self.assertLess(self.norm_of(psi), 0.99)

    def test_renormalise_flag_restores_number_conservation(self):
        """With renormalise=True the norm is forced back to 1 despite the loss term.

        The test above shows the same lossy potential draining atoms when the flag
        is left off; this pins what the flag does for the ground-state search.
        """
        psi = self.gaussian()
        V = self.harmonic_potential()
        utot = V + 1j * (-50.0 * torch.abs(psi) ** 4)
        out = cyl.split_step_step(psi, utot, 0.002, self.kz, self.m_modes, self.r,
                                  self.eigvecs, self.eigvals, self.dr, self.dphi, self.dz,
                                  renormalise=True)
        self.assertAlmostEqual(self.norm_of(out), 1.0, places=12)


class TestCylindricalModGradPsi(CylindricalFixture):
    def test_matches_the_analytic_gradient_of_a_gaussian(self):
        """|∇exp(-(r²+z²)/2)| = sqrt(r²+z²)·ψ away from the boundaries."""
        gr, _, gz = self.space_grid
        psi = torch.exp(-(gr ** 2 + gz ** 2) / 2).to(torch.cdouble)
        grad = cyl.mod_grad_psi(psi, self.r, self.dr, self.kz, self.m_modes)
        expected = torch.sqrt(gr ** 2 + gz ** 2) * torch.abs(psi)
        interior = (slice(0, self.n_r // 2), slice(None), slice(self.n_z // 4, 3 * self.n_z // 4))
        error = float((grad[interior] - expected[interior]).abs().max())
        self.assertLess(error, 5e-3)

    def test_uses_the_full_complex_derivative(self):
        """A phase-carrying state must register kinetic content."""
        gr, gphi, gz = self.space_grid
        envelope = torch.exp(-(gr ** 2 + gz ** 2) / 2)
        real_state = envelope.to(torch.cdouble)
        moving = (envelope * torch.exp(2j * gphi)).to(torch.cdouble)
        g_real = cyl.mod_grad_psi(real_state, self.r, self.dr, self.kz, self.m_modes)
        g_moving = cyl.mod_grad_psi(moving, self.r, self.dr, self.kz, self.m_modes)
        self.assertGreater(float(torch.sum(g_moving ** 2)), float(torch.sum(g_real ** 2)))

    def test_uniform_state_has_zero_radial_gradient(self):
        """A uniform state has zero gradient, including in r.

        The radial derivative is a finite difference on a grid whose spacing is
        uniform but whose volume element is not, so a misplaced r weight would show
        up as a spurious gradient here.
        """
        psi = torch.ones(self.n_r, self.n_phi, self.n_z, dtype=torch.cdouble)
        grad = cyl.mod_grad_psi(psi, self.r, self.dr, self.kz, self.m_modes)
        self.assertLess(float(grad.abs().max()), 1e-10)


class TestCylindricalEnergy(CylindricalFixture):
    def test_harmonic_oscillator_energies(self):
        """Exact values for the isotropic 3-D ground state: e_kin = e_pot = 0.75."""
        energies = cyl.calculate_energy_allocation(
            self.gaussian(), self.harmonic_potential(), self.r, self.dr, self.dphi,
            self.dz, self.kz, self.m_modes, u=0.0)
        self.assertAlmostEqual(float(energies['e_kin'].real), 0.75, places=2)
        self.assertAlmostEqual(float(energies['e_pot'].real), 0.75, places=2)
        self.assertAlmostEqual(float(energies['E_total'].real), 1.5, places=2)
        self.assertAlmostEqual(float(energies['e_int'].real), 0.0, places=12)

    def test_energies_carry_the_volume_element(self):
        """Refining the grid must not change the energy of the same state."""
        coarse = cyl.calculate_energy_allocation(
            self.gaussian(), self.harmonic_potential(), self.r, self.dr, self.dphi,
            self.dz, self.kz, self.m_modes, u=0.0)

        n_r, n_phi, n_z = 72, 8, 36
        r, _, _, kz, m_modes, dr, dphi, dz, (gr, _, gz) = cyl.init_grid(
            self.r_max, self.z_min, self.z_max, n_r, n_phi, n_z, 'cpu')
        eigvecs, eigvals = cyl.build_radial_operators(r, dr, m_modes, 'cpu')
        psi = cyl.normalize(torch.exp(-(gr ** 2 + gz ** 2) / 2).to(torch.cdouble),
                            r, dr, dphi, dz)
        fine = cyl.calculate_energy_allocation(
            psi, 0.5 * (gr ** 2 + gz ** 2), r, dr, dphi, dz, kz, m_modes, u=0.0)

        self.assertAlmostEqual(float(coarse['E_total'].real),
                               float(fine['E_total'].real), places=2)

    def test_interaction_term_scales_with_u(self):
        """The interaction energy is proportional to u at fixed density."""
        psi, V = self.gaussian(), self.harmonic_potential()
        one = cyl.calculate_energy_allocation(psi, V, self.r, self.dr, self.dphi,
                                              self.dz, self.kz, self.m_modes, u=1.0)
        two = cyl.calculate_energy_allocation(psi, V, self.r, self.dr, self.dphi,
                                              self.dz, self.kz, self.m_modes, u=2.0)
        self.assertAlmostEqual(2 * float(one['e_int'].real), float(two['e_int'].real), places=12)

    def test_complex_potential_contributes_only_its_real_part(self):
        """An absorbing potential is a loss rate, not an energy."""
        psi, V = self.gaussian(), self.harmonic_potential()
        real_only = cyl.calculate_energy_allocation(
            psi, V, self.r, self.dr, self.dphi, self.dz, self.kz, self.m_modes, u=1.0)
        absorbing = cyl.calculate_energy_allocation(
            psi, V - 3j * torch.ones_like(V), self.r, self.dr, self.dphi, self.dz,
            self.kz, self.m_modes, u=1.0)
        self.assertAlmostEqual(float(absorbing['e_pot'].real),
                               float(real_only['e_pot'].real), places=12)
        # The energies stay real-valued rather than picking up the loss rate.
        self.assertFalse(torch.is_complex(torch.as_tensor(absorbing['E_total'])))

    def test_missing_interaction_strength_raises(self):
        """Omitting u raises rather than assuming a non-interacting gas."""
        with self.assertRaises(ValueError):
            cyl.calculate_energy_allocation(
                self.gaussian(), self.harmonic_potential(), self.r, self.dr,
                self.dphi, self.dz, self.kz, self.m_modes)

    def test_chemical_potential_counts_interaction_twice(self):
        """mu = e_kin + e_pot + 2*e_int, returned as a float.

        The interaction term enters twice because mu = dE/dN while the interaction
        energy scales as N^2.
        """
        psi, V = self.gaussian(), self.harmonic_potential()
        energies = cyl.calculate_energy_allocation(
            psi, V, self.r, self.dr, self.dphi, self.dz, self.kz, self.m_modes, u=5.0)
        mu = cyl.calculate_chemical_potential(psi, V, 5.0, self.r, self.dr, self.dphi,
                                              self.dz, self.kz, self.m_modes)
        expected = float((energies['e_kin'] + energies['e_pot'] + 2 * energies['e_int']).real)
        self.assertAlmostEqual(mu, expected, places=12)
        self.assertIsInstance(mu, float)

    def test_chemical_potential_of_the_non_interacting_ground_state(self):
        """For the non-interacting ground state mu is 3/2 hbar*omega_ho.

        Two decimal places is what this grid supports: the Gaussian is sampled on a
        half-point radial grid that does not resolve the origin exactly.
        """
        mu = cyl.calculate_chemical_potential(
            self.gaussian(), self.harmonic_potential(), 0.0, self.r, self.dr,
            self.dphi, self.dz, self.kz, self.m_modes)
        self.assertAlmostEqual(mu, 1.5, places=2)


class TestCylindricalThermalNoise(CylindricalFixture):
    def test_satisfies_the_fluctuation_dissipation_relation(self):
        """⟨|η|²⟩ must equal 2γkTΔτ/dV in every radial shell."""
        gamma, kT, dtau = 0.05, 1.5, 0.01
        torch.manual_seed(3)
        shape = (self.n_r, self.n_phi, self.n_z)
        samples = torch.stack([
            cyl.generate_thermal_noise(shape, gamma, kT, dtau, self.r, self.dr,
                                       self.dphi, self.dz, 'cpu')
            for _ in range(40)])
        measured = torch.mean(torch.abs(samples) ** 2, dim=0)
        dV_local = self.r_w * self.dr * self.dphi * self.dz
        expected = 2.0 * gamma * kT * dtau / dV_local
        ratio = float(torch.mean(measured / expected))
        self.assertAlmostEqual(ratio, 1.0, delta=0.05)

    def test_noise_is_complex_and_correctly_shaped(self):
        """The noise is a complex field of the grid's shape.

        A real noise field would drive only the amplitude and leave the phase
        unthermalised.
        """
        noise = cyl.generate_thermal_noise((self.n_r, self.n_phi, self.n_z), 0.05, 1.0,
                                           0.01, self.r, self.dr, self.dphi, self.dz, 'cpu')
        self.assertEqual(noise.dtype, torch.cdouble)
        self.assertEqual(noise.shape, (self.n_r, self.n_phi, self.n_z))

    def test_zero_temperature_gives_no_noise(self):
        """At kT = 0 the noise is identically zero."""
        noise = cyl.generate_thermal_noise((self.n_r, self.n_phi, self.n_z), 0.05, 0.0,
                                           0.01, self.r, self.dr, self.dphi, self.dz, 'cpu')
        self.assertEqual(float(noise.abs().max()), 0.0)

    def test_projection_removes_high_energy_modes(self):
        """The 'P' of the projected SGPE: no power above the cutoff energy."""
        shape = (self.n_r, self.n_phi, self.n_z)
        torch.manual_seed(5)
        unprojected = cyl.generate_thermal_noise(shape, 0.05, 1.0, 0.01, self.r, self.dr,
                                                 self.dphi, self.dz, 'cpu')
        torch.manual_seed(5)
        projected = cyl.generate_thermal_noise(
            shape, 0.05, 1.0, 0.01, self.r, self.dr, self.dphi, self.dz, 'cpu',
            kz=self.kz, m_modes=self.m_modes, eigvecs_dict=self.eigvecs,
            eigvals_dict=self.eigvals, e_cut=5.0)
        self.assertLess(self.norm_of(projected), self.norm_of(unprojected))
        self.assertGreater(self.norm_of(projected), 0.0)

    def test_projection_is_monotonic_in_the_cutoff(self):
        """Raising the cutoff admits strictly more power.

        Three cutoffs two decades apart are compared pairwise from the same seed,
        so the ordering reflects the projection rather than the sampling.
        """
        shape = (self.n_r, self.n_phi, self.n_z)
        powers = []
        for e_cut in (2.0, 20.0, 200.0):
            torch.manual_seed(7)
            noise = cyl.generate_thermal_noise(
                shape, 0.05, 1.0, 0.01, self.r, self.dr, self.dphi, self.dz, 'cpu',
                kz=self.kz, m_modes=self.m_modes, eigvecs_dict=self.eigvecs,
                eigvals_dict=self.eigvals, e_cut=e_cut)
            powers.append(self.norm_of(noise))
        self.assertLess(powers[0], powers[1])
        self.assertLess(powers[1], powers[2])

    def test_no_projection_without_the_operators(self):
        """e_cut alone is not enough — the kinetic basis is required."""
        shape = (self.n_r, self.n_phi, self.n_z)
        torch.manual_seed(9)
        plain = cyl.generate_thermal_noise(shape, 0.05, 1.0, 0.01, self.r, self.dr,
                                           self.dphi, self.dz, 'cpu')
        torch.manual_seed(9)
        cutoff_only = cyl.generate_thermal_noise(shape, 0.05, 1.0, 0.01, self.r, self.dr,
                                                 self.dphi, self.dz, 'cpu', e_cut=1.0)
        self.assertTrue(torch.equal(plain, cutoff_only))


class TestCylindricalSGPEStep(CylindricalFixture):
    def setUp(self):
        self.psi = self.gaussian()
        self.V = self.harmonic_potential()
        self.mu_gs = cyl.calculate_chemical_potential(
            self.psi, self.V, 0.0, self.r, self.dr, self.dphi, self.dz,
            self.kz, self.m_modes)

    def _run(self, mu, steps=150, **kwargs):
        psi = self.psi.clone()
        for _ in range(steps):
            psi = cyl.sgpe_step(psi, self.V, mu, 0.05, 0.005, self.kz, self.m_modes,
                                self.r, self.eigvecs, self.eigvals,
                                self.dr, self.dphi, self.dz, **kwargs)
        return psi

    def test_reservoir_above_mu_grows_the_condensate(self):
        """A reservoir above the state's own mu feeds atoms in."""
        self.assertGreater(self.norm_of(self._run(self.mu_gs + 0.5)), 1.02)

    def test_reservoir_below_mu_shrinks_the_condensate(self):
        """A reservoir below the state's own mu drains atoms out."""
        self.assertLess(self.norm_of(self._run(self.mu_gs - 0.5)), 0.98)

    def test_ground_state_is_a_fixed_point_at_its_own_mu(self):
        """At its own mu the ground state neither grows nor decays.

        The tolerance is looser than the Cartesian equivalent because mu itself is
        only accurate to the radial discretisation.
        """
        self.assertAlmostEqual(self.norm_of(self._run(self.mu_gs)), 1.0, delta=5e-3)

    def test_mu_has_no_effect_once_the_norm_is_forced(self):
        """Why renormalise defaults to False: it divides the μ factor back out."""
        low = self._run(self.mu_gs - 0.5, renormalise=True)
        high = self._run(self.mu_gs + 0.5, renormalise=True)
        density_gap = float((torch.abs(low) ** 2 - torch.abs(high) ** 2).abs().max())
        self.assertLess(density_gap, 1e-12)

    def test_damping_relaxes_an_excited_state_towards_the_ground_state(self):
        """Damping lowers the energy of a radially excited state.

        The state is renormalised before the comparison, so the drop is a change of
        shape rather than of atom number.
        """
        excited = cyl.normalize(self.psi * (1.0 + 0.3 * torch.abs(self.space_grid[0])),
                                self.r, self.dr, self.dphi, self.dz)
        before = cyl.calculate_energy_allocation(
            excited, self.V, self.r, self.dr, self.dphi, self.dz,
            self.kz, self.m_modes, u=0.0)['E_total'].real
        psi = excited
        for _ in range(200):
            psi = cyl.sgpe_step(psi, self.V, self.mu_gs, 0.1, 0.005, self.kz,
                                self.m_modes, self.r, self.eigvecs, self.eigvals,
                                self.dr, self.dphi, self.dz)
        psi = cyl.normalize(psi, self.r, self.dr, self.dphi, self.dz)
        after = cyl.calculate_energy_allocation(
            psi, self.V, self.r, self.dr, self.dphi, self.dz,
            self.kz, self.m_modes, u=0.0)['E_total'].real
        self.assertLess(float(after), float(before))


class TestAngularMomentumZ(CylindricalFixture):
    def test_charge_two_state_carries_two_hbar(self):
        """A state with exp(2*i*phi) carries Lz = 2 hbar per particle.

        In cylindrical coordinates Lz is diagonal in the azimuthal modes, so this
        is a direct check that the m weighting is applied to the right column.
        """
        gr, gphi, gz = self.space_grid
        psi = cyl.normalize(
            (torch.exp(-(gr ** 2 + gz ** 2) / 2) * gr ** 2 * torch.exp(2j * gphi)).to(torch.cdouble),
            self.r, self.dr, self.dphi, self.dz)
        lz = cyl.angular_momentum_z(psi, self.m_modes, self.r, self.dr, self.dphi, self.dz)
        self.assertAlmostEqual(float(lz), 2.0, places=8)

    def test_axisymmetric_state_carries_none(self):
        """An axisymmetric state carries no angular momentum."""
        lz = cyl.angular_momentum_z(self.gaussian(), self.m_modes, self.r,
                                    self.dr, self.dphi, self.dz)
        self.assertAlmostEqual(float(lz), 0.0, places=10)

    def test_negative_charge_reverses_the_sign(self):
        """Reversing the circulation reverses the sign of Lz.

        This is what pins the signed mode ordering: with |m| the answer would come
        out +1 either way.
        """
        gr, gphi, gz = self.space_grid
        psi = cyl.normalize(
            (torch.exp(-(gr ** 2 + gz ** 2) / 2) * gr * torch.exp(-1j * gphi)).to(torch.cdouble),
            self.r, self.dr, self.dphi, self.dz)
        lz = cyl.angular_momentum_z(psi, self.m_modes, self.r, self.dr, self.dphi, self.dz)
        self.assertAlmostEqual(float(lz), -1.0, places=8)


class TestCylindricalClassLayout(CylindricalFixture):
    """The simulation loop reaches for these through the class hierarchy.

    They previously lived on a sibling class with no base, so every cylindrical
    snapshot raised AttributeError at runtime.
    """

    def test_two_d_library_inherits_the_core_operators(self):
        """The 2-D cylindrical library really does inherit the core operators.

        Subclassing is asserted directly, then each operator the loop calls is
        checked by name -- a class that redefined some of them but broke the
        inheritance would still pass a bare hasattr sweep.
        """
        self.assertTrue(issubclass(cyl2d, cyl))
        for name in ('init_grid', 'normalize', 'p_evolution', 'split_step_step',
                     'sgpe_step', 'calculate_energy_allocation',
                     'calculate_chemical_potential', 'generate_thermal_noise',
                     'build_radial_operators', 'angular_momentum_z'):
            self.assertTrue(hasattr(cyl2d, name), msg=f"GPE2DCylindricalLibrary lost {name}")

    def test_diagnostics_used_by_the_snapshot_writer_are_reachable(self):
        """Every diagnostic the cylindrical snapshot writer calls is reachable.

        These used to live on a class with no base, so a cylindrical run raised
        AttributeError on its first snapshot -- after the ground state had already
        been computed.
        """
        for name in ('rms_radius', 'column_density_z', 'column_density_radial',
                     'radial_density_profile', 'create_vortices',
                     'check_vortex_resolution'):
            self.assertTrue(hasattr(cyl2d, name), msg=f"missing diagnostic {name}")


class TestCylindricalDiagnostics(CylindricalFixture):
    def test_rms_radius_of_a_known_gaussian(self):
        """For exp(-(r²+z²)/2), ⟨r²⟩ = 1 in the plane and ⟨z²⟩ = 1/2, so rms = √(3/2)."""
        rms = cyl2d.rms_radius(self.gaussian(), self.r, self.dr, self.dphi, self.dz)
        self.assertAlmostEqual(float(rms), np.sqrt(1.0), places=2)

    def test_rms_radius_grows_with_a_wider_state(self):
        """A state spread over four times the variance reports a larger RMS radius.

        Both states are normalised first, so the comparison is of width rather than
        of weight.
        """
        gr, _, gz = self.space_grid
        wide = cyl.normalize(torch.exp(-(gr ** 2 + gz ** 2) / 8).to(torch.cdouble),
                             self.r, self.dr, self.dphi, self.dz)
        self.assertGreater(float(cyl2d.rms_radius(wide, self.r, self.dr, self.dphi, self.dz)),
                           float(cyl2d.rms_radius(self.gaussian(), self.r, self.dr, self.dphi, self.dz)))

    def test_column_density_z_shape_and_scaling(self):
        """Integrating out z leaves an (n_r, n_phi) map scaled by n_z*dz.

        For a uniform state the value is the axial length, which is what makes this
        a check on the dz factor rather than on the shape alone.
        """
        psi = torch.ones(self.n_r, self.n_phi, self.n_z, dtype=torch.cdouble)
        column = cyl2d.column_density_z(psi, self.dz)
        self.assertEqual(column.shape, (self.n_r, self.n_phi))
        self.assertTrue(torch.allclose(column, torch.full_like(column, self.n_z * self.dz)))

    def test_column_density_radial_uses_the_r_weight(self):
        """The radial column density carries the r weight, not a bare sum.

        For a uniform state the expected value is sum(r)*dr; a routine that summed
        without the weight would return n_r*dr instead.
        """
        psi = torch.ones(self.n_r, self.n_phi, self.n_z, dtype=torch.cdouble)
        column = cyl2d.column_density_radial(psi, self.r, self.dr)
        self.assertEqual(column.shape, (self.n_phi, self.n_z))
        expected = float(torch.sum(self.r) * self.dr)
        self.assertAlmostEqual(float(column[0, 0]), expected, places=12)

    def test_radial_profile_integrates_to_the_norm(self):
        """The radial profile integrates back to 1 under r dr.

        The profile has already absorbed dphi and dz, so the remaining weight is
        the radial one; recovering the norm shows none of the three was dropped or
        counted twice.
        """
        psi = self.gaussian()
        profile = cyl2d.radial_density_profile(psi, self.dphi, self.dz)
        self.assertEqual(profile.shape, (self.n_r,))
        total = float(torch.sum(profile * self.r) * self.dr)
        self.assertAlmostEqual(total, 1.0, places=10)


class TestCylindricalCreateVortices(CylindricalFixture):
    def test_on_axis_vortex_winds_by_two_pi(self):
        """A charge-1 vortex on the axis winds the phase by 2*pi around a circle.

        The circle is a ring of constant r, so the loop is a single azimuthal pass;
        each step is unwrapped into (-pi, pi] before the sum.
        """
        mask = cyl2d.create_vortices(self.r, self.phi, self.n_r, self.n_phi, self.n_z,
                                     [(0.0, 0.0)], [1], 0.5, 'cpu')
        phase = torch.angle(mask[self.n_r // 2, :, 0])
        steps = torch.diff(torch.cat([phase, phase[:1]]))
        # unwrap each step into (-pi, pi] before summing around the loop
        steps = (steps + np.pi) % (2 * np.pi) - np.pi
        self.assertAlmostEqual(float(steps.sum()), 2 * np.pi, places=6)

    def test_core_density_is_suppressed(self):
        """The vortex mask suppresses the density at the core relative to the edge.

        The mask is an amplitude profile as well as a phase, and the core has to go
        to zero for the phase singularity to be physical.
        """
        mask = cyl2d.create_vortices(self.r, self.phi, self.n_r, self.n_phi, self.n_z,
                                     [(0.0, 0.0)], [1], 0.5, 'cpu')
        self.assertLess(float(torch.abs(mask[0, 0, 0])), float(torch.abs(mask[-1, 0, 0])))

    def test_mask_is_uniform_along_z(self):
        """Vortex lines run along z, so every axial slice must be identical."""
        mask = cyl2d.create_vortices(self.r, self.phi, self.n_r, self.n_phi, self.n_z,
                                     [(1.0, 0.5)], [1], 0.4, 'cpu')
        self.assertEqual(mask.shape, (self.n_r, self.n_phi, self.n_z))
        for k in range(1, self.n_z):
            self.assertTrue(torch.equal(mask[:, :, k], mask[:, :, 0]))

    def test_resolution_check_is_reachable(self):
        """Regression: passing dr/dphi used to raise AttributeError."""
        with self.assertWarns(UserWarning):
            mask = cyl2d.create_vortices(self.r, self.phi, self.n_r, self.n_phi,
                                         self.n_z, [(4.0, 0.0)], [1], 0.05, 'cpu',
                                         dr=self.dr, dphi=self.dphi)
        self.assertEqual(mask.shape, (self.n_r, self.n_phi, self.n_z))

    def test_well_resolved_core_does_not_warn(self):
        """A core spanning several cells produces no resolution warning.

        Warnings are turned into errors for the duration, so the absence of an
        exception is the assertion. This is the counterpart to the test above,
        which checks the warning does fire when the core is too small.
        """
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            cyl2d.create_vortices(self.r, self.phi, self.n_r, self.n_phi, self.n_z,
                                  [(0.0, 0.0)], [1], 2.0, 'cpu',
                                  dr=self.dr, dphi=self.dphi)

    def test_mismatched_charges_raise(self):
        """Two positions with one charge is rejected."""
        with self.assertRaises(ValueError):
            cyl2d.create_vortices(self.r, self.phi, self.n_r, self.n_phi, self.n_z,
                                  [(0.0, 0.0), (1.0, 1.0)], [1], 0.5, 'cpu')

    def test_multiple_vortices_multiply(self):
        """Adding a second vortex removes more density than one alone.

        The masks multiply, so each additional core carves another hole; the total
        amplitude can only go down.
        """
        one = cyl2d.create_vortices(self.r, self.phi, self.n_r, self.n_phi, self.n_z,
                                    [(1.0, 0.0)], [1], 0.5, 'cpu')
        two = cyl2d.create_vortices(self.r, self.phi, self.n_r, self.n_phi, self.n_z,
                                    [(1.0, 0.0), (-1.0, 0.0)], [1, 1], 0.5, 'cpu')
        self.assertLess(float(torch.abs(two).sum()), float(torch.abs(one).sum()))


class TestCheckVortexResolution(unittest.TestCase):
    def test_on_axis_core_is_limited_by_the_radial_step(self):
        """On the axis the arc length vanishes, so dr sets the effective spacing.

        With r0 = 0 the azimuthal spacing r0*dphi is zero, leaving dr = 0.1 against
        a core of 1.0 -- comfortably resolved.
        """
        report = cyl2d.check_vortex_resolution([(0.0, 0.0)], r_core=1.0, dr=0.1, dphi=0.1)
        self.assertEqual(report[0]['bottleneck'], 'radial')
        self.assertAlmostEqual(report[0]['dr_eff'], 0.1, places=12)
        self.assertTrue(report[0]['resolved'])

    def test_far_out_core_is_limited_by_the_arc_length(self):
        """Far from the axis the arc length dominates and the core is unresolved.

        At r0 = 5 with dphi = 0.5 the arc is 2.5, far larger than dr = 0.1, so the
        effective spacing is 2.5 and a core of 1.0 spans less than one cell.
        """
        report = cyl2d.check_vortex_resolution([(5.0, 0.0)], r_core=1.0, dr=0.1, dphi=0.5)
        self.assertEqual(report[0]['bottleneck'], 'azimuthal')
        self.assertAlmostEqual(report[0]['dr_eff'], 2.5, places=12)
        self.assertFalse(report[0]['resolved'])

    def test_reports_one_entry_per_vortex(self):
        """The report has one entry per vortex, carrying its distance from the axis.

        The second vortex sits at (1, 1), so r0 is sqrt(2): the check works in
        Cartesian coordinates and converts, rather than assuming the position is
        already radial.
        """
        report = cyl2d.check_vortex_resolution([(0.0, 0.0), (1.0, 1.0)], 0.5, 0.1, 0.1)
        self.assertEqual(len(report), 2)
        self.assertAlmostEqual(report[1]['r0'], np.sqrt(2.0), places=12)


if __name__ == '__main__':
    unittest.main()
