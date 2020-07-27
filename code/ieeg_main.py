
import ieeg_utilities as ieeg
import numpy as np

patient_id = ['ID07', 'ID07', 'ID07', 'ID08', 'ID08', 'ID08', 'ID11', 'ID11', 'ID11', 'ID11', 'ID11', 'ID11']
time_begin = [[32, 7], [35, 15], [38, 22], [57, 58], [60, 10], [64, 40], [60, 5], [62, 12], [65, 0], [129, 48],
              [132, 35], [136, 35]]
duration = 60

for i in range(len(patient_id)):
    if time_begin[i][1] > 9:
        name = 'fc_' + patient_id[i] + '_' + str(time_begin[i][0]) + 'h' + str(time_begin[i][1]) + 'min'
    else:
        name = 'fc_' + patient_id[i] + '_' + str(time_begin[i][0]) + 'h' + str(time_begin[i][1]) + 'min'

    r2 = np.load('../data/' + name + '_r2.npy')
    r2_dt = np.load('../data/' + name + '_r2_dt.npy')
    h2 = np.load('../data/' + name + '_h2.npy')
    h2_dt = np.load('../data/' + name + '_h2_dt.npy')
    print(name + ': ' + str(np.mean(np.abs(r2))))

    #ieeg.plot_corr_connectivity(r2, r2_dt, h2, h2_dt, save_name=name)
    #ieeg.make_corr_connectivity(patient_id[i], time_begin[i], duration, t_shift=0.05, plot_name=patient_id[i] + '_0')

