import src.ieeg_utilities as ieeg
from scipy.io import loadmat
import numpy as np
import json
from tqdm import tqdm
from joblib import Parallel, delayed
import os
import matplotlib.pyplot as plt


def resample_signal(signal, dt):
    """
    Resample time series data if needed (iEEG is given with dt of 2ms).
    Args:
        signal: time series data.
        dt: Desired timestep of iEEG.

    Returns:

    """
    new_signal = []
    for i in range(len(signal)):
        new_signal.append(np.interp(np.arange(0, len(signal[i]), dt / 2.), np.arange(0, len(signal[i])), signal[i]))
    return np.asarray(new_signal)


def plot_output(index, targets, volts, duration, unclamped_neurons, base_output_folder='./outputs'):

    if not os.path.exists(base_output_folder):
        os.mkdir(base_output_folder)

    plt.close('all')
    fig, ax = plt.subplots(unclamped_neurons, 1, figsize=(16, 12))
    for i in range(unclamped_neurons):
        ax[i].plot(volts[-unclamped_neurons + i, :], color='darkred')
        ax[i].plot(targets[-unclamped_neurons + i, :duration], color='k', alpha=0.5, linestyle='--')
    plt.savefig(base_output_folder + '/series_' + str(index) + '.png')


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

    patient_id = [str(i) for i in range(370, 0, -10)]
    time_begin = [[0, 0]]
    duration = 8
    t_lag = 0.7

    experiment_duration = 4000  # /fs = ms
    n_targets = 10

    network_dt = 1.0
    experiment_duration = int(experiment_duration * 2 / network_dt)
    #
    # dataset_name = '../data/model_outputs/EEG_set.json'
    # dataset_id = 1
    # # load dataset
    # with open(dataset_name, 'r') as infile:
    #     data = json.load(infile)['patient_' + str(dataset_id)]
    # input_data = np.array(data['input']) / 2.0
    # print(f"Input iEEG signals: {input_data.shape}")
    #
    # target_data = np.array(data['target']) / 2.0
    # print(f"Target iEEG signals: {target_data.shape}")
    #
    # if network_dt != 2.0:
    #     print('Upsampling signals...')
    #     target_data = resample_signal(target_data, network_dt)
    #     print('...Done.')


    def lin_corr(i, n_jobs):
        network_target_data = np.load(f"../data/fully_recurrent_outputs/all_target_volts_{patient_id[i]}.npy")
        plot_output(patient_id[i], network_target_data[-2 * n_targets:-n_targets, :], network_target_data[-n_targets:, :], experiment_duration, n_targets, "../doc/figures/")

        info_mat = {"fs": 1000}

        data_mat = {"EEG": network_target_data[: -n_targets, :]}  # all
        ieeg.lin_corr(patient_id[i] + "_output", data_mat, info_mat, time_begin=time_begin[0], duration=duration, t_lag=t_lag, critical_corr=0.4, pbar_pos=i % n_jobs)

        data_mat = {"EEG": network_target_data[-2 * n_targets: -n_targets, :]}  # only outputs
        ieeg.lin_corr(patient_id[i] + "_output", data_mat, info_mat, time_begin=time_begin[0], duration=duration, t_lag=t_lag, critical_corr=0.4, pbar_pos=i % n_jobs)

        data_mat = {"EEG": network_target_data[-n_targets:, :]}
        ieeg.lin_corr(patient_id[i] + "_target", data_mat, info_mat, time_begin=time_begin[0], duration=duration, t_lag=t_lag, critical_corr=0.4, pbar_pos=i % n_jobs)

    n_jobs = 1
    #Parallel(n_jobs=n_jobs)(delayed(lin_corr)(i, n_jobs) for i in range(len(patient_id)))
    for i in range(len(patient_id)):
        lin_corr(i, n_jobs)
        break

        # ieeg.plot_corr_connectivity(r2, r2_dt, h2, h2_dt, save_name=name)
    # ieeg.make_corr_connectivity(patient_id[i], time_begin[i], duration, t_shift=0.05, plot_name=patient_id[i] + '_0')

