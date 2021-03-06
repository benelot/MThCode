import utilities_train as utrain
import utilities_figures as ufig
import utilities_various as uvar
import os

if __name__ == '__main__':

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'

    ids_all = []
    pre = 'single_'
    for attempt in range(3):
        print('------------------------------ ' + 'Attempt Nr. ' + str(attempt) + ' ------------------------------')
        post = '_' + str(attempt)

        # params_change = [[pre + 'ID07_32h07m' + post, 'ID07', [32, 7], 'beginning'],
        #                  [pre + 'ID07_35h15m' + post, 'ID07', [35, 15], 'middle'],
        #                  [pre + 'ID07_38h22m' + post, 'ID07', [38, 22], 'end'],
        params_change = [[pre + 'ID08_57h00m' + post, 'ID08', [57, 00], 'before'],
                         [pre + 'ID08_60h10m' + post, 'ID08', [60, 10], 'middle'],
                         [pre + 'ID08_64h40m' + post, 'ID08', [64, 40], 'end']]
                         # [pre + 'ID11a_60h05m' + post, 'ID11', [60, 5], 'beginning'],
                         # [pre + 'ID11a_62h12m' + post, 'ID11', [62, 12], 'middle'],
                         # [pre + 'ID11a_65h00m' + post, 'ID11', [65, 0], 'end'],
                         # [pre + 'ID11b_129h48m_bs001_' + post, 'ID11', [129, 48], 'beginning'],
                         # [pre + 'ID11b_132h35m_bs001_' + post, 'ID11', [132, 35], 'middle'],
                         # [pre + 'ID11b_136h35m_bs001_' + post, 'ID11', [136, 35], 'end']]

        ids_attempt = []
        for i, val in enumerate(params_change):
            print('(M) Status: Train model: ' + val[0])
            ids_attempt.append(val[0])
            ids_all.append(val[0])

            params = {'id_': ids_all[-1],
                      'model_type': 'single_layer',  # To be removed
                      'path2data': '../data/',
                      'patient_id': val[1],
                      'time_begin': val[2],  # [hour, minute]
                      'duration': 10,  # seconds
                      'brain_state': val[3],
                      'add_id': '(M)',
                      # model parameters ------------------------
                      'visible_size': 'all',  # 'all' or scalar
                      'hidden_size': 0,  # improve: portion
                      'lambda': 0,
                      'af': 'relu',  # 'relu', 'linear', 'sigmoid'
                      'bias': True,
                      'window_size': 0,
                      'resample': 512,
                      # train parameters -------------------------
                      'loss_function': 'mae',  # 'mse' or 'mae'
                      'lr': 0.0002,
                      'batch_size': 5,
                      'shuffle': False,
                      'normalization': 'standard_positive',  # 'min_max', 'standard', None
                      'epochs': 23}

            utrain.train_and_test(params)
            ufig.plot_train_test(ids_all[-1], [1, 3, 5, 7])

        ufig.plot_multi_boxplots(ids=ids_attempt, x='batch_size', y='correlation', hue='brain_state',
                                 save_name=pre + 'corr' + post, ylim=(0, 1))
        #ufig.plot_multi_boxplots(ids=ids_attempt, x='patient_id', y='mae', hue='brain_state', save_name=pre + 'mae' + post)
        #ufig.plot_multi_boxplots(ids=ids_attempt, x='patient_id', y='mse', hue='brain_state', save_name=pre + 'mse' + post)

    ufig.mean_weights(ids=ids_all, save_name=pre)
