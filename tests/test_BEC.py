import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append(".")
from src.models import BEC as BEC

bec = BEC.BEC()

imprint_position_x = np.array([[[0],[0],[0]]])
imprint_position_y = np.array([[[0],[0],[0]]])
imprinting_charge = np.array([[[1],[1],[1]]])
imprint_times = np.array([[7,10,15]])

vortex_array = bec._create_vortex_list(imprint_position_x, imprint_position_y, imprinting_charge, imprint_times)
print(vortex_array)

bec._calculate_all_phases(vortex_array)

imprint_position_x = np.array([[[0,0],[0]]])
imprint_position_y = np.array([[[0,0],[0]]])
imprinting_charge = np.array([[[1,2],[1]]])
imprint_times = np.array([[7,9]])

vortex_array = bec._create_vortex_list(imprint_position_x, imprint_position_y, imprinting_charge, imprint_times)
print(vortex_array)

bec._calculate_all_phases(vortex_array)

print(f"{imprint_times}")