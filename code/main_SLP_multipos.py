import utilities_train as utrain
import utilities_figures as ufig
import utilities_various as uvar
import os

if __name__ == '__main__':

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'

    ids_all = []
    pre = 'multiple_pos_'
    for attempt in range(3):
        print('------------------------------ ' + 'Attempt Nr. ' + str(attempt) + ' ------------------------------')
        post = '_' + str(attempt)

        params_change = [[pre + 'ID08_57h00m' + post, 'ID07', [32, 10], 'NREM beginning', 150],
                         [pre + 'ID08_57h58m' + post, 'ID08', [58, 0], 'NREM beginning', 122],
                         [pre + 'ID08_60h10m' + post, 'ID08', [60, 2], 'NREM middle', 122],
                         [pre + 'ID08_60h58m' + post, 'ID08', [64, 40], 'NREM end', 122],
                         [pre + 'ID08_64h40m' + post, 'ID11', [60, 5], 'NREM beginning', 64],
                         [pre + 'ID08_65h50m' + post, 'ID11', [62, 10], 'NREM middle', 64]]

        ids_attempt = []
        for i, val in enumerate(params_change):
            print('(SLP) ----- Status: Train model: ' + val[0])
            ids_attempt.append(val[0])
            ids_all.append(val[0])

            params = {'id_': ids_attempt[-1],
                      'model_type': 'single_layer',  # None=SRNN, single_layer=SLP
                      'path2data': '../data/',
                      'patient_id': val[1],
                      'time_begin': val[2],  # [hour, minute]
                      'artificial_signal': [False, False],  # [bool on/off, bool small_weights]
                      'duration': 6*60,  # seconds
                      'brain_state': val[3],
                      'add_id': '(SLP)',
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

            #utrain.train_and_test(params)
            #ufig.plot_train_test(params['id_'], n_nodes='all')
            #if params['id_'] == 'SLP_ID11b_136h30m_0':
                #ufig.plot_prediction('SLP_ID11b_136h30m_0', n_nodes=5)

    ufig.mean_weights(ids=ids_all, save_name=pre)
    #ufig.plot_performance(ids=ids_all, save_name=pre)
