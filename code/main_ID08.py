import utilities_train as utrain
import utilities_figures as ufig
import utilities_various as uvar
import os

if __name__ == '__main__':

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'

    pre = 'allpos'

    ids = []
    h_offset = 57
    for h_ in range(9):
        for m in range(30):
            h = h_ + h_offset
            m = 2 * m

            zero = ''
            if m < 10:
                zero = '0'

            t_string = str(h) + 'h' + zero + str(m) + 'm'
            patient_id = 'ID08'

            params = {'id_': pre + '_' + patient_id + '_' + t_string,
                      'model_type': None,  # To be removed
                      'path2data': '../data/',
                      'patient_id': patient_id,
                      'time_begin': [h, m],  # [hour, minute]
                      'artificial_signal': [False, False],  # [bool on/off, bool small_weights]
                      'duration': 120,  # seconds
                      'brain_state': t_string,
                      'add_id': '(' + patient_id + ')',
                      # model parameters ------------------------
                      'visible_size': 'all',  # 'all' or scalar
                      'hidden_size': 120,  # improve: portion
                      'lambda': 0,
                      'af': 'relu',  # 'relu', 'linear', 'sigmoid'
                      'bias': True,
                      'window_size': 50,
                      'resample': 256,
                      # train parameters -------------------------
                      'loss_function': 'mae',  # 'mse' or 'mae'
                      'lr': 0.001,
                      'batch_size': 1024,
                      'shuffle': True,
                      'weight_decay': 0.0001,
                      'normalization': 'all_standard_positive',  # 'min_max', 'standard', None
                      'epochs': 180}

            print('Status: Training ' + params['id_'])
            # utrain.train_and_test(params)
            node_idx = [k for k in range(20)]
            ufig.plot_train_test(params['id_'], n_nodes=15, node_idx=node_idx)
            # ids.append(params['id_'])

# ufig.plot_multi_boxplots(ids=ids, x='brain_state', y='correlation', save_name=pre + 'corr',
#                          ylim=(0, 1))
# ufig.plot_multi_boxplots(ids=ids, x='brain_state', y='mae', save_name=pre + 'mae')
# ufig.plot_multi_boxplots(ids=ids, x='brain_state', y='mse', save_name=pre + 'mse')
#
# ufig.mean_weights(ids=ids, save_name=pre)
