import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append(".")
from src.utils import setup_simulations as ss

# Test save_parameters_to_json
params = {'x':np.array([1,2,3]), 'y':np.array([[1,2],[3,4]]), 'z':5, 't': 'potential'}
ss.save_parameters_to_json(params)