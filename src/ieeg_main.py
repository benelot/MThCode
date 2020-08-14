import src.ieeg_utilities as ieeg
from scipy.io import loadmat
import numpy as np
import json

if __name__ == '__main__':
    # patient_id = ['ID11']
    # time_begin = [[55, 5], ]
    # duration = 60 # 61
    # t_lag = 0.7 # 15
    #
    # for i in range(len(patient_id)):
    #     print(str(i))
    #     # Load and prepare data
    #     data_mat = loadmat('../data/' + patient_id[i] + '_' + str(time_begin[0]) + 'h.mat')
    #     info_mat = loadmat('../data/' + patient_id[i] + '_info.mat')
    #     ieeg.lin_corr(patient_id[i], data_mat, info_mat, time_begin=time_begin[i], duration=duration[i], t_lag=t_lag)
    #     break

    patient_id = ["0"]
    time_begin = [[0, 0]]
    duration = 8
    t_lag = 0.7

    dataset_name = '../data/model_outputs/EEG_set.json'
    dataset_id = 1
    experiment_duration = 4000  # /fs ms

    network_dt = 1.0
    experiment_duration = int(experiment_duration * 2 / network_dt)

    # load dataset
    with open(dataset_name, 'r') as infile:
        data = json.load(infile)['patient_' + str(dataset_id)]
    input_data = np.array(data['input']) / 2.0
    print(f"Input iEEG signals: {input_data.shape}")

    target_data = np.array(data['target']) / 2.0
    print(f"Target iEEG signals: {target_data.shape}")

    for i in range(len(patient_id)):
        data_mat = {"EEG": np.load(f"../data/model_outputs/volts_{i}.npy").transpose()}
        data_mat["EEG"] = np.vstack((data_mat["EEG"], target_data[:, :experiment_duration]))
        info_mat = {"fs": 1000}
        ieeg.lin_corr(patient_id[i], data_mat, info_mat, time_begin=time_begin[i], duration=duration, t_lag=t_lag, critical_corr=0.1)

    # ieeg.plot_corr_connectivity(r2, r2_dt, h2, h2_dt, save_name=name)
    # ieeg.make_corr_connectivity(patient_id[i], time_begin[i], duration, t_shift=0.05, plot_name=patient_id[i] + '_0')

