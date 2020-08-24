import ieeg_utilities as ieeg
from scipy.io import loadmat
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

patient_id = ['ID07', 'ID07', 'ID07', 'ID08', 'ID08', 'ID08', 'ID11', 'ID11', 'ID11', 'ID11', 'ID11', 'ID11']
time_begin = [[32, 7], [35, 15], [38, 22], [57, 58], [60, 10], [64, 40], [60, 5], [62, 12], [65, 0], [129, 48],
              [132, 35], [136, 35]]
duration = 30
i = 0
data_raw = []
# Load and prepare data
for i in range(len(patient_id)):
    data_mat = loadmat('../data/' + patient_id[i] + '_' + str(time_begin[i][0]) + 'h.mat')
    info_mat = loadmat('../data/' + patient_id[i] + '_info.mat')
    fs = float(info_mat['fs'])
    sample_begin = int(time_begin[i][1] * 60 * fs)
    sample_end = sample_begin + int(duration * fs)
    data_raw.append(data_mat['EEG'][:, sample_begin:sample_end].transpose())

# data_all = (data_raw - np.mean(data_raw)) / (5 * np.std(data_raw))
# data_all = data_all / 2 + 0.5
#
# data_each = (data_raw - np.mean(data_raw, axis=0)) / (5 * np.std(data_raw, axis=0))
# data_each = data_each / 2 + 0.5

ieeg.plot_distribution(data_raw, qq_plot=False, xlim=(0, 1), title='all')
# ieeg.plot_distribution(data_each, qq_plot=False, xlim=(0, 1), title='each')

#plot_distribution(data: np.ndarray, xlim: tuple=None, n_clusters: int=6, qq_plot: bool=True, title: str=''):


# for i in range(len(patient_id)):
#     print(str(i))
#
#     ieeg.lin_corr(patient_id=patient_id[i], time_begin=time_begin[i], duration=duration)
#     break



# ieeg.plot_corr_connectivity(r2, r2_dt, h2, h2_dt, save_name=name)
# ieeg.make_corr_connectivity(patient_id[i], time_begin[i], duration, t_shift=0.05, plot_name=patient_id[i] + '_0')

