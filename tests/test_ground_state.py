import sys
import unittest

import torch

sys.path.append('.')

from src.library.ground_state import GroundState
from src.library.ground_state_cylindrical import CylindricalGroundState
from src.library.gpe_cylindrical_library import GPECylindricalLibrary as cyl


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
        """Scaffold a regression test for exact hand-computed scalar diagnostics."""
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
        """Scaffold a regression test for exact hand-computed scalar diagnostics."""
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


if __name__ == '__main__':
    unittest.main()