import utilities_train as utrain
import utilities_figures as ufig
import utilities_various as uvar

from scipy.stats import mode
from scipy import signal
import numpy as np
from scipy.io import loadmat
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import os

if __name__ == '__main__':

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'

    ids = []
    pre = 'SLP_allpos_'
    patient_id = 'ID11a'
    h_offset = 59
    h_range = 7
    corrmean = []

    for h_ in range(h_range):
        for m in range(10):  # 30
            h = h_ + h_offset
            m = 6 * m  # 2

            zero = ''
            if m < 10:
                zero = '0'

            t_string = str(h) + 'h' + zero + str(m) + 'm'
            load_patient_id = patient_id
            if 'ID11' in patient_id:
                load_patient_id = 'ID11'

            params = {'id_': pre + patient_id + '_' + t_string,
                      'model_type': 'single_layer',  # None=SRNN, single_layer=SLP
                      'path2data': '../data/',
                      'patient_id': load_patient_id,
                      'time_begin': [h, m],  # [hour, minute]
                      'artificial_signal': [False, False],  # [bool on/off, bool small_weights]
                      'duration': 6 * 60,  # seconds
                      'brain_state': 'None',
                      'add_id': '(All_' + patient_id + ')',
                      # model parameters ------------------------
                      'visible_size': 'all',  # 'all' or scalar
                      'hidden_size': 0,  # improve: portion
                      'lambda': 0.0,
                      'af': 'relu',  # 'relu', 'linear', 'sigmoid'
                      'bias': True,
                      'window_size': 30,
                      'resample': 256,
                      # train parameters -------------------------
                      'loss_function': 'mae',  # 'mse' or 'mae'
                      'lr': 0.001,
                      'batch_size': 1024,
                      'shuffle': True,
                      'weight_decay': 0.0001,
                      'normalization': 'all_standard_positive',  # 'min_max', 'standard', None
                      'epochs': 100}

            ids.append(params['id_'])
            data_mat = loadmat(
                params['path2data'] + params['patient_id'] + '_' + str(params['time_begin'][0]) + 'h.mat')
            info_mat = loadmat(params['path2data'] + params['patient_id'] + '_' + 'info.mat')
            fs = float(info_mat['fs'])
            sample_begin = int(np.round(params['time_begin'][1] * 60 * fs))
            sample_end = int(np.round(sample_begin + params['duration'] * fs))
            data = data_mat['EEG'][:, sample_begin:sample_end].transpose()
            data = signal.resample(data, num=int(data.shape[0] / fs * params['resample']), axis=0)

            # Correlation
            corr_mat = np.corrcoef(data.T)
            corr_val = np.mean(np.abs(corr_mat), axis=0)
            corrmean.append(corr_val)
            print('Done: ' + params['id_'])

    np.save('../data/corrmean_ID11_59to65h_all.npy', np.asarray(corrmean))
