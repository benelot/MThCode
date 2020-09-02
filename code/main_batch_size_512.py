import utilities_train as utrain
import utilities_figures as ufig
import utilities_various as uvar
import os

if __name__ == '__main__':

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'

    ids_all = []
    pre = 'SLP_5min_'
    for attempt in range(3):
        print('------------------------------ ' + 'Attempt Nr. ' + str(attempt) + ' ------------------------------')
        post = '_' + str(attempt)

        params_change = [[pre + 'ID07_32h10m' + post, 'ID07', [32, 10], 'NREM beginning'],
                         [pre + 'ID07_35h10m' + post, 'ID07', [35, 10], 'NREM middle'],
                         [pre + 'ID07_38h15m' + post, 'ID07', [38, 15], 'NREM end'],
                         [pre + 'ID08_58h25m' + post, 'ID08', [58, 25], 'NREM beginning'],
                         [pre + 'ID08_60h08m' + post, 'ID08', [60, 8], 'NREM middle'],
                         [pre + 'ID08_64h40m' + post, 'ID08', [64, 40], 'NREM end'],
                         [pre + 'ID11a_60h05m' + post, 'ID11', [60, 5], 'NREM beginning'],
                         [pre + 'ID11a_62h10m' + post, 'ID11', [62, 10], 'NREM middle'],
                         [pre + 'ID11a_65h00m' + post, 'ID11', [65, 0], 'NREM end'],
                         [pre + 'ID11b_129h45m' + post, 'ID11', [129, 45], 'NREM beginning'],
                         [pre + 'ID11b_132h20m' + post, 'ID11', [132, 20], 'NREM middle'],
                         [pre + 'ID11b_136h30m' + post, 'ID11', [136, 30], 'NREM end']]

        ids_attempt = []
        for i, val in enumerate(params_change):
            print('E ----- Status: Train model: ' + val[0])
            ids_attempt.append(val[0])
            ids_all.append(val[0])

            params = {'id_': ids_attempt[-1],
                      'model_type': 'single_layer',  # To be removed
                      'path2data': '../data/',
                      'patient_id': val[1],
                      'time_begin': val[2],  # [hour, minute]
                      'artificial_signal': [False, False],  # [bool on/off, bool small_weights]
                      'duration': 5*60,  # seconds
                      'brain_state': val[3],
                      'add_id': '(E)',
                      # model parameters ------------------------
                      'visible_size': 'all',  # 'all' or scalar
                      'hidden_size': 0,  # improve: portion
                      'lambda': 0,
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
            ufig.plot_train_test(ids_attempt[-1], n_nodes=15)

        ufig.plot_multi_boxplots(ids=ids_attempt, x='patient_id', y='correlation', hue='brain_state', save_name=pre + 'corr' + post)
        ufig.plot_multi_boxplots(ids=ids_attempt, x='patient_id', y='mae', hue='brain_state', save_name=pre + 'mae' + post)
        ufig.plot_multi_boxplots(ids=ids_attempt, x='patient_id', y='mse', hue='brain_state', save_name=pre + 'mse' + post)

    #ufig.mean_weights(ids=ids_all, save_name=pre)
