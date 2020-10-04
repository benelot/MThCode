import ieeg_utilities as ieeg
from scipy.io import loadmat
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# patient_ids = ['ID07', 'ID07', 'ID07',
#                'ID08', 'ID08', 'ID08',
#                'ID11', 'ID11', 'ID11',
#                'ID11', 'ID11', 'ID11']
# time_begins = [[32, 10], [35, 10], [38, 15],
#                [58, 0], [60, 2], [64, 40],
#                [60, 5], [62, 10], [64, 55],
#                [129, 45], [133, 36], [136, 30]]
#
# ieeg.determine_fc_change(patient_id=patient_ids, time_begin=time_begins, max_sample_size=60*60, t_length=240, save_name='fc_change')


#ieeg.determine_sample_size(patient_id=patient_ids, time_begin=time_begins, max_sample_size=600, dt=5, load_name='final', save_name='final1')
# ieeg.plot_corr_connectivity(r2, r2_dt, h2, h2_dt, save_name=name)
# ieeg.make_corr_connectivity(patient_id[i], time_begin[i], duration, t_shift=0.05, plot_name=patient_id[i] + '_0')

