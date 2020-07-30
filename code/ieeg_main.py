
import ieeg_utilities as ieeg
import numpy as np

patient_id = ['ID07', 'ID07', 'ID07', 'ID08', 'ID08', 'ID08', 'ID11', 'ID11', 'ID11', 'ID11', 'ID11', 'ID11']
time_begin = [[32, 0], [35, 0], [38, 0], [57, 0], [60, 0], [64, 0], [60, 0], [62, 12], [65, 0], [129, 48],
              [132, 35], [136, 35]]

duration = 60
corr_change = []
t = []
k = [32, 35, 38]
for _, val in enumerate(k):
    for i in range(20):
        print(str(i))
        # cc, ts, tau, cctl = ieeg.lin_corr(patient_id[i], time_begin[i], duration, t_lag=.7)
        # ieeg.lin_corr_plot(cc, ts, tau, patient_id[i], time_begin[i])
        corr_dt, t_size = ieeg.determine_sample_size('ID07', [val, i * 3], max_sample_size=180, dt=5)
        corr_change.append(corr_dt)
        t.append(t_size)

ieeg.plot_determine_sample_size(corr_change, t, save_name='default1')






    # ieeg.plot_corr_connectivity(r2, r2_dt, h2, h2_dt, save_name=name)
    # ieeg.make_corr_connectivity(patient_id[i], time_begin[i], duration, t_shift=0.05, plot_name=patient_id[i] + '_0')

