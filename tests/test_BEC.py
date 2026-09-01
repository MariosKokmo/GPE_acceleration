import sys
import unittest
from unittest.mock import MagicMock

sys.path.append(".")
import numpy as np
import torch
from src.models import BEC as BEC


def _make_bec(n1=32, n2=4, n3=32):
    """Construct a BEC with lightweight stand-ins for the app and system.

    The BEC constructor now requires (parameters, system, app, simulation_name).
    The vortex-list / phase helpers under test only need a CPU device, a logger,
    the real-space axes and the grid resolution, so we supply minimal stubs.
    """
    app = MagicMock()
    app.device = torch.device("cpu")
    # app.logger and app.time are auto-created MagicMocks (info/error are no-ops)

    system = MagicMock()
    x1 = torch.linspace(-5.0, 5.0, n1, dtype=torch.float64)
    x2 = torch.linspace(-1.0, 1.0, n2, dtype=torch.float64)
    x3 = torch.linspace(-5.0, 5.0, n3, dtype=torch.float64)
    system.space_axes = (x1, x2, x3)
    system.simulation_parameters = {"Grid_resolution": [n1, n2, n3]}

    return BEC.BEC(parameters={}, system=system, app=app, simulation_name="test")


class TestBEC(unittest.TestCase):
    def test_create_vortex_list_single(self):
        bec = _make_bec()

        # One simulation (outer list of length 1) with three imprint times,
        # each imprinting a single charge-1 vortex at the origin.
        imprint_position_x = np.array([[[0], [0], [0]]])
        imprint_position_y = np.array([[[0], [0], [0]]])
        imprinting_charge = np.array([[[1], [1], [1]]])
        imprint_times = np.array([7, 10, 15])  # 1-D list of times

        vortex_array = bec._create_vortex_list(
            imprint_position_x, imprint_position_y, imprinting_charge, imprint_times
        )
        self.assertIsNotNone(vortex_array)
        self.assertEqual(set(vortex_array.keys()), {7, 10, 15})

        bec._calculate_all_phases(vortex_array)
        # Phases are cached by (x, y, charge), so identical imprints collapse
        # to a single entry. Expect exactly the unique keys.
        expected_keys = {
            (tuple(v[0]), tuple(v[1]), tuple(v[2])) for v in vortex_array.values()
        }
        self.assertEqual(set(bec.all_phases.keys()), expected_keys)

    def test_create_vortex_list_multiple(self):
        bec = _make_bec()

        # One simulation with two imprint times: the first imprints two
        # vortices, the second imprints one. Plain lists keep the ragged
        # per-imprint structure (numpy >= 1.24 rejects ragged arrays).
        imprint_position_x = [[[0, 0], [0]]]
        imprint_position_y = [[[0, 0], [0]]]
        imprinting_charge = [[[1, 2], [1]]]
        imprint_times = [7, 9]

        vortex_array = bec._create_vortex_list(
            imprint_position_x, imprint_position_y, imprinting_charge, imprint_times
        )
        self.assertIsNotNone(vortex_array)
        self.assertEqual(set(vortex_array.keys()), {7, 9})
        self.assertEqual(vortex_array[7].shape, (3, 2))
        self.assertEqual(vortex_array[9].shape, (3, 1))

        bec._calculate_all_phases(vortex_array)
        # Phases are cached by (x, y, charge), so identical imprints collapse
        # to a single entry. Expect exactly the unique keys.
        expected_keys = {
            (tuple(v[0]), tuple(v[1]), tuple(v[2])) for v in vortex_array.values()
        }
        self.assertEqual(set(bec.all_phases.keys()), expected_keys)


class TestBECPhaseOps(unittest.TestCase):
    """Regression tests for phase helpers.

    These paths call CommonUtils helpers (update_phase / extract_phase) that
    were previously, incorrectly, invoked on GPELibrary and raised
    AttributeError at runtime.
    """

    def setUp(self):
        self.bec = _make_bec()
        self.shape = (32, 4, 32)
        self.bec.psi = torch.ones(self.shape, dtype=torch.cdouble)

    def test_repetitive_imprint_applies_phase(self):
        phase = torch.full(self.shape, 0.5, dtype=torch.float64)
        before = self.bec.psi.clone()
        self.bec._repetitive_imprint(phase)
        expected = before * torch.exp(1j * phase)
        self.assertTrue(torch.allclose(self.bec.psi, expected))

    def test_extract_phase_runs(self):
        self.bec.psi = torch.exp(
            1j * torch.full(self.shape, 0.3, dtype=torch.float64)
        ).to(torch.cdouble)
        phase = self.bec._extract_phase()
        self.assertEqual(phase.shape, self.shape)
        self.assertTrue(torch.all(torch.isfinite(phase)))

    def test_imprint_vortices_runs(self):
        vortices = np.array([[0], [0], [1]])  # one charge-1 vortex at the origin
        self.bec._imprint_vortices(vortices)
        self.assertEqual(self.bec.psi.shape, self.shape)
        self.assertTrue(torch.all(torch.isfinite(self.bec.psi)))


if __name__ == "__main__":
    unittest.main()
