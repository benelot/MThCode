from scipy.io import loadmat
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib import animation, rc
from IPython.display import HTML, Image
import ieeg_processing as ieeg
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

# Import signal
"""
model_type = [['in', '57h', '58m', int(58*60*1024), int(1*60*1024)],
              ['in', '60h', '14m', int(14*60*1024), int(1*60*1024)],
              ['in', '64h', '40m', int(40*60*1024), int(1*60*1024)],
              ['is', '57h', '58m', int(58*60*1024), int(1*60*1024)],
              ['is', '60h', '14m', int(14*60*1024), int(1*60*1024)],
              ['is', '64h', '40m', int(40*60*1024), int(1*60*1024)]]

ieeg.emp_connectome(patient_ID='07', hour='29to39_all', sperseg=120, soverlap=20)
"""
# Load data
data_mat = loadmat('../data/ID07_35h.mat')
info_mat = loadmat('../data/ID07_info.mat')
fs = info_mat['fs']
sample_begin = int(5*60*fs)
sample_end = sample_begin + int(1*fs)
data_raw = data_mat['EEG'][:, sample_begin:sample_end].transpose()

# Normalizing
ssc = StandardScaler()
data_ssc = ssc.fit_transform(data_raw)

mmsc = MinMaxScaler(feature_range=(-1, 1))
data_mmsc = mmsc.fit_transform(data_raw)

data_global_ssc = (data_raw - np.mean(data_raw))/np.std(data_raw)

X = data_raw
X_std = (X - np.min(X)) / (np.max(X) - np.min(X))
data_global_mmsc = X_std * (1 - (-1) + (-1))

#ieeg.plot_distribution(data_raw, n_clusters=5, qq_plot=False, title='Raw')
#ieeg.plot_distribution(data_ssc, n_clusters=5, qq_plot=False, title='Standard_single_feature')
#ieeg.plot_distribution(data_mmsc, n_clusters=5, qq_plot=False, title='MinMax_single_feature')
#ieeg.plot_distribution(data_global_ssc, n_clusters=5, qq_plot=False, title='Standard_all_feature')
ieeg.plot_distribution(data_global_mmsc, n_clusters=5, qq_plot=False, title='MinMax_all_feature')
