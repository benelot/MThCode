import utilities_train as utrain
import utilities_figures as ufig
import utilities_various as uvar
import os

if __name__ == '__main__':
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'

    pre = 'val_'

    params_change = [[pre + 'SLP', 'single_layer', 50, 0, 0],
                     [pre + 'SRNN_L00', None, 50, 0, 0],
                     [pre + 'SRNN_L02', None, 30, 2, 0.2]]

    for i, val in enumerate(params_change):
        params = {'id_': val[0],
                  'model_type': val[1],  # None=SRNN, single_layer=SLP
                  'path2data': '../data/',
                  'patient_id': None,
                  'time_begin': None,  # [hour, minute]
                  'artificial_signal': [True, False],  # [bool on/off, bool small_weights]
                  'duration': 0,  # seconds
                  'brain_state': 'validation',
                  'add_id': '(V)',
                  # model parameters ------------------------
                  'visible_size': val[2],  # 'all' or scalar
                  'hidden_size': val[3],  # improve: portion
                  'lambda': val[4],
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
                  'epochs': 250}

        utrain.train_and_test(params)
        ufig.plot_train_test(params['id_'], n_nodes=15)
