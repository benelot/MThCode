import ieeg_utilities as ieeg
from scipy.io import loadmat
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

"""
params_change = [[pre + 'ID07_32h10m' + post, 'ID07', [32, 10], 'NREM beginning'],
                 [pre + 'ID07_35h10m' + post, 'ID07', [35, 10], 'NREM middle'],
                 [pre + 'ID07_38h15m' + post, 'ID07', [38, 15], 'NREM end'],
                 [pre + 'ID08_58h25m' + post, 'ID08', [58, 25], 'NREM beginning'],
                 [pre + 'ID08_60h08m' + post, 'ID08', [60, 8], 'NREM middle'],
                 [pre + 'ID08_64h40m' + post, 'ID08', [64, 40], 'NREM end'],
                 [pre + 'ID11a_60h05m' + post, 'ID11', [60, 5], 'NREM beginning'],
                 [pre + 'ID11a_62h10m' + post, 'ID11', [62, 10], 'NREM middle'],
                 [pre + 'ID11a_65h00m' + post, 'ID11', [65, 0], 'NREM end'],
                 [pre + 'ID11b_129h45m' + post, 'ID11', [129, 45], 'NREM beginning'],
                 [pre + 'ID11b_133h30m' + post, 'ID11', [133, 30], 'NREM middle'],
                 [pre + 'ID11b_136h30m' + post, 'ID11', [136, 30], 'NREM end']]


# Load and prepare data
data_mat = loadmat('../data/' + patient_id + '_' + str(time_begin[0]) + 'h.mat')
info_mat = loadmat('../data/' + patient_id + '_info.mat')
fs = float(info_mat['fs'])
sample_begin = int(time_begin[1] * 60 * fs)
sample_end = sample_begin + int(duration * fs)
data_raw = data_mat['EEG'][:, sample_begin:sample_end].transpose()

plt.figure(figsize=(5, 5))
sns.pairplot()
"""
ieeg.lin_corr(patient_id='ID07', time_begin=[32, 10], duration=float(5*60), t_lag=1, critical_corr=0.4)


# ieeg.plot_corr_connectivity(r2, r2_dt, h2, h2_dt, save_name=name)
# ieeg.make_corr_connectivity(patient_id[i], time_begin[i], duration, t_shift=0.05, plot_name=patient_id[i] + '_0')

