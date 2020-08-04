
import ieeg_utilities as ieeg
import numpy as np

patient_id = ['ID07', 'ID07', 'ID07', 'ID08', 'ID08', 'ID08', 'ID11', 'ID11', 'ID11', 'ID11', 'ID11', 'ID11']
time_begin = [[32, 0], [35, 0], [38, 0], [58, 0], [60, 0], [64, 30], [60, 0], [62, 0], [65, 0], [129, 40],
              [132, 30], [136, 30]]

ieeg.determine_sample_size(patient_id, time_begin, max_sample_size=10*60, dt=5)


# ieeg.plot_corr_connectivity(r2, r2_dt, h2, h2_dt, save_name=name)
# ieeg.make_corr_connectivity(patient_id[i], time_begin[i], duration, t_shift=0.05, plot_name=patient_id[i] + '_0')

