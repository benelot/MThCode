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
"""

data_mat = loadmat('../data/ID07_32h.mat')
info_mat = loadmat('../data/ID07_info.mat')
fs = float(info_mat['fs'])
sample_begin = int(np.round(10 * 60 * fs))
sample_end = int(np.round(sample_begin + 5 * 60 * fs))
data = data_mat['EEG'][:, sample_begin:sample_end].transpose()
data_mat = []

ieeg.plot_distribution(data)


# ieeg.plot_corr_connectivity(r2, r2_dt, h2, h2_dt, save_name=name)
# ieeg.make_corr_connectivity(patient_id[i], time_begin[i], duration, t_shift=0.05, plot_name=patient_id[i] + '_0')

