import utilities_train as utrain
import utilities_figures as ufig
import utilities_various as uvar
import os

if __name__ == '__main__':

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'

    ids = []
    pre = 'SLP_allpos_nWD_'
    patient_id = 'ID11b'
    h_offset = 129
    h_range = 9

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
                      'weight_decay': 0,
                      'normalization': 'all_standard_positive',  # 'min_max', 'standard', None
                      'epochs': 100}

            ids.append(params['id_'])
            utrain.train_and_test(params)


