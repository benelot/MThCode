import utilities_train as utrain
import utilities_figures as ufig
import utilities_various as uvar
import os

if __name__ == '__main__':

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'

    ids_all = []
    pre = 'SRNN_L00_varHidden_wWD_'
    for attempt in range(1):
        print('------------------------------ ' + 'Attempt Nr. ' + str(attempt) + ' ------------------------------')
        post = '_' + str(attempt)

        params_change = [[pre + '0' + post, 'ID07', [32, 10], 'Nr. Hidden: 0', 0],
                         [pre + '30' + post, 'ID07', [32, 10], 'Nr. Hidden: 30', 20],
                         [pre + '90' + post, 'ID07', [32, 10], 'Nr. Hidden: 90', 120],
                         [pre + '120' + post, 'ID07', [32, 10], 'Nr. Hidden: 150', 150]]

        ids_attempt = []
        for i, val in enumerate(params_change):
            print('(vH) ----- Status: Train model: ' + val[0])
            ids_attempt.append(val[0])
            ids_all.append(val[0])

            params = {'id_': ids_attempt[-1],
                      'model_type': None,  # None=SRNN, single_layer=SLP
                      'path2data': '../data/',
                      'patient_id': val[1],
                      'time_begin': val[2],  # [hour, minute]
                      'artificial_signal': [False, False],  # [bool on/off, bool small_weights]
                      'duration': 6*60,  # seconds
                      'brain_state': val[3],
                      'add_id': '(vH)',
                      # model parameters ------------------------
                      'visible_size': 'all',  # 'all' or scalar
                      'hidden_size': val[4],  # improve: portion
                      'lambda': 0.0,
                      'af': 'relu',  # 'relu', 'linear', 'sigmoid'
                      'bias': True,
                      'window_size': 8,
                      'resample': 256,
                      # train parameters -------------------------
                      'loss_function': 'mae',  # 'mse' or 'mae'
                      'lr': 0.001,
                      'batch_size': 1024,
                      'shuffle': True,
                      'weight_decay': 0.00005,
                      'normalization': 'all_standard_positive',  # 'min_max', 'standard', None
                      'epochs': 100}

            utrain.train_and_test(params)
            ufig.plot_train_test(params['id_'], n_nodes='all')

    #ufig.mean_weights(ids=ids_all, save_name=pre)
    ufig.plot_performance(ids=ids_all, save_name=pre)
