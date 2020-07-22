from scipy.io import loadmat
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
import pyinform as pyinform
import pwlf

import ieeg_utilities as ieeg

patient_id = 'ID11'
time_begin = [62, 12]
duration = 20
ieeg.make_corr_connectivity(patient_id, time_begin, duration, t_shift=0.1, plot_name=patient_id + '_1')

patient_id = 'ID11'
time_begin = [65, 0]
duration = 20
ieeg.make_corr_connectivity(patient_id, time_begin, duration, t_shift=0.1, plot_name=patient_id + '_2')
