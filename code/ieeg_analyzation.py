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

import ieeg_processing as ieeg

# Load data
data_mat = loadmat('../data/ID11_60h.mat')
info_mat = loadmat('../data/ID11_info.mat')
fs = info_mat['fs']
sample_begin = int(5*60*fs)
sample_end = sample_begin + int(60*fs)
data_raw = data_mat['EEG'][:, sample_begin:sample_end].transpose()

x = data_raw[:10000, 0]
y = data_raw[:10000, 1]
t_shift = int(0.5*fs)  # samples


h2, r, h2_x_to_y, r_x_to_y, h2_shift, r_shift = ieeg.correlation_metrics(x, y, t_shift=t_shift)

plt.figure()
plt.plot(h2_shift), plt.grid(), plt.show()

plt.figure()
plt.plot(r_shift), plt.grid(), plt.show()

print('h2: ' + str(h2) + ' shift: ' + str(h2_x_to_y))
print('r: ' + str(r) + ' shift: ' + str(r_x_to_y))
