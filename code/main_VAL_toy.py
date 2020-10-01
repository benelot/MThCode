import utilities_train as utrain
import utilities_figures as ufig
import utilities_various as uvar
import os

if __name__ == '__main__':
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'

    pre = 'val_toy_'

    params_change = [[pre + 'strong', [True, False]],
                     [pre + 'weak', [True, True]]]

    for i, val in enumerate(params_change):
        params = {'id_': val[0],
                  'model_type': 'single_layer',  # None=SRNN, single_layer=SLP
                  'path2data': '../data/',
                  'patient_id': None,
                  'time_begin': None,  # [hour, minute]
                  'artificial_signal': val[1],  # [bool on/off, bool small_weights]
                  'duration': 60,  # seconds
                  'brain_state': 'validation',
                  'add_id': '(V)',
                  # model parameters ------------------------
                  'visible_size': 'all',  # 'all' or scalar
                  'hidden_size': 0,  # improve: portion
                  'lambda': 0,
                  'af': 'relu',  # 'relu', 'linear', 'sigmoid'
                  'bias': True,
                  'window_size': 30,
                  'resample': 512,
                  # train parameters -------------------------
                  'loss_function': 'mae',  # 'mse' or 'mae'
                  'lr': 0.001,
                  'batch_size': 1024,
                  'shuffle': True,
                  'weight_decay': 0.01,
                  'normalization': 'all_standard_positive',  # 'min_max', 'standard', None
                  'epochs': 100}

        #utrain.train_and_test(params)
        #ufig.plot_train_test(params['id_'], n_nodes='all')
        ufig.plot_weights(params['id_'])
