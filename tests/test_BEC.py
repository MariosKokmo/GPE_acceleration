import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import unittest
sys.path.append(".")
from src.models import BEC as BEC
import src.library.gpe_library as gpe
import src.library.ground_state as gs
import src.utils.read_write_utils as rw
from src.utils import video_creation
import numpy as np
import os
import torch
from pathlib import Path
from sys import platform


class TestBEC(unittest.TestCase):
    def test_create_vortex_list_single(self):
        bec = BEC.BEC()

        imprint_position_x = np.array([[[0], [0], [0]]])
        imprint_position_y = np.array([[[0], [0], [0]]])
        imprinting_charge = np.array([[[1], [1], [1]]])
        imprint_times = np.array([[7, 10, 15]])

        vortex_array = bec._create_vortex_list(imprint_position_x, imprint_position_y, imprinting_charge, imprint_times)
        self.assertIsNotNone(vortex_array)  # Example assertion
        print(vortex_array)
        bec._calculate_all_phases(vortex_array)
        print(bec.all_phases)

    def test_create_vortex_list_multiple(self):
        bec = BEC.BEC()

        imprint_position_x = np.array([[[0, 0], [0]]])
        imprint_position_y = np.array([[[0, 0], [0]]])
        imprinting_charge = np.array([[[1, 2], [1]]])
        imprint_times = np.array([[7, 9]])

        vortex_array = bec._create_vortex_list(imprint_position_x, imprint_position_y, imprinting_charge, imprint_times)
        self.assertIsNotNone(vortex_array)  # Example assertion
        print(vortex_array)

        bec._calculate_all_phases(vortex_array)
        print(bec.all_phases)

if __name__ == "__main__":
    unittest.main()