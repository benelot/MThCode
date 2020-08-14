import utilities_train as utrain
import utilities_figures as ufig
import utilities_various as uvar
import os

if __name__ == '__main__':

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'

    ids_all = []
    pre = 'wd_multipos_'
    for attempt in range(3):
        print('------------------------------ ' + 'Attempt Nr. ' + str(attempt) + ' ------------------------------')
        post = '_' + str(attempt)

        time_begin = [[57, 0], [57, 30], [58, 0], [58, 30], [60, 0], [60, 30], [60, 55], [64, 20], [64, 40],
                      [65, 10], [65, 50]]

        params_change = []
        for k, time in enumerate(time_begin):
            if time[1] < 10:
                in_fill = '0'
            else:
                in_fill = ''
            time_label = str(time[0]) + 'h' + in_fill + str(time[1]) + 'm'
            params_change.append([pre + 'ID08_' + time_label + post, time, time_label])

        ids_attempt = []
        for i, val in enumerate(params_change):
            print('(M) Status: Train model: ' + val[0])
            ids_attempt.append(val[0])
            ids_all.append(val[0])

            params = {'id_': ids_all[-1],
                      'model_type': 'single_layer',  # To be removed
                      'path2data': '../data/',
                      'patient_id': 'ID08',
                      'time_begin': val[1],  # [hour, minute]
                      'artificial_signal': [False, False],  # [bool on/off, bool small_weights]
                      'duration': 100,  # seconds
                      'brain_state': val[2],
                      'add_id': '(M)',
                      # model parameters ------------------------
                      'visible_size': 'all',  # 'all' or scalar
                      'hidden_size': 0,  # improve: portion
                      'lambda': 0,
                      'af': 'relu',  # 'relu', 'linear', 'sigmoid'
                      'bias': True,
                      'window_size': 0,
                      'resample': 256,
                      # train parameters -------------------------
                      'loss_function': 'mae',  # 'mse' or 'mae'
                      'lr': 0.001,
                      'batch_size': 512,
                      'shuffle': True,
                      'weight_decay': 0.0001,
                      'normalization': 'all_standard_positive',  # 'min_max', 'standard', None
                      'epochs': 150}

            utrain.train_and_test(params)
            ufig.plot_train_test(ids_all[-1], n_nodes=15)

        ufig.plot_multi_boxplots(ids=ids_attempt, x='batch_size', y='correlation', hue='brain_state',
                                 save_name=pre + 'corr' + post, ylim=(0, 1))
        ufig.plot_multi_boxplots(ids=ids_attempt, x='patient_id', y='mae', hue='brain_state', save_name=pre + 'mae' + post)
        ufig.plot_multi_boxplots(ids=ids_attempt, x='patient_id', y='mse', hue='brain_state', save_name=pre + 'mse' + post)

    ufig.mean_weights(ids=ids_all, save_name=pre)
