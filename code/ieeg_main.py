import ieeg_utilities as ieeg
from scipy.io import loadmat
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

data_mat_ID11 = loadmat('../data/ID11_all_fs64.mat')
data_mat_ID12 = loadmat('../data/ID12_all_fs64.mat')
fs = 64

sns.set_style('white')
fig = plt.figure(figsize=(8, 8))
gs = fig.add_gridspec(2, 1)

ax0 = fig.add_subplot(gs[:, :2])


sns.set_style('white')
fig = plt.figure(figsize=(8, 8))
gs = fig.add_gridspec(2, 1)

# ieeg.plot_corr_connectivity(r2, r2_dt, h2, h2_dt, save_name=name)
# ieeg.make_corr_connectivity(patient_id[i], time_begin[i], duration, t_shift=0.05, plot_name=patient_id[i] + '_0')

