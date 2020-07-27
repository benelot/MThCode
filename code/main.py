import utilities_train as utrain
import utilities_figures as ufig
import os

if __name__ == '__main__':

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'

    ids_all = []
    pre = 'linear_'
    for attempt in range(2):
        print('------------------------------ ' + 'Attempt Nr. ' + str(attempt) + ' ------------------------------')
        post = '_' + str(attempt)

        params_change = [[pre + 'ID07_32h07m' + post, 'ID07', [32, 7]],
                         [pre + 'ID07_35h15m' + post, 'ID07', [35, 15]],
                         [pre + 'ID07_38h22m' + post, 'ID07', [38, 22]],
                         [pre + 'ID08_57h58m' + post, 'ID08', [57, 58]],
                         [pre + 'ID08_60h10m' + post, 'ID08', [60, 10]],
                         [pre + 'ID08_64h40m' + post, 'ID08', [64, 40]],
                         [pre + 'ID11a_60h05m' + post, 'ID11', [60, 5]],
                         [pre + 'ID11a_62h12m' + post, 'ID11', [62, 12]],
                         [pre + 'ID11a_65h00m' + post, 'ID11', [65, 0]],
                         [pre + 'ID11b_129h48m' + post, 'ID11', [129, 48]],
                         [pre + 'ID11b_132h35m' + post, 'ID11', [132, 35]],
                         [pre + 'ID11b_136h35m' + post, 'ID11', [136, 35]]]

        ids_attempt = []
        for i, val in enumerate(params_change):
            ids_attempt.append(val[0])
            ids_all.append(val[0])

            params = {'id_': ids_attempt[-1],
                      'model_type': None,  # To be removed
                      'path2data': '../data/',
                      'patient_id': val[1],
                      'time_begin': val[2],  # [hour, minute]
                      'duration': 60,  # seconds
                      # model parameters ------------------------
                      'visible_size': 'all',  # 'all' or scalar
                      'hidden_size': 0,  # improve: portion
                      'lambda': 0,
                      'af': 'linear',  # 'relu', 'linear', 'sigmoid'
                      'bias': True,
                      'window_size': 30,
                      # train parameters -------------------------
                      'loss_function': 'mae',  # 'mse' or 'mae'
                      'lr': 0.0001,
                      'normalization': None,  # 'min_max', 'standard', None
                      'epochs': 400}

            utrain.train_and_test(params)
            utrain.test_train_set(ids_attempt[-1])
            ufig.plot_train_test(ids_attempt[-1], [3, 8, 13, 17], lim_nr_samples=2000, predict_train_set=True)

        ufig.plot_multi_boxplots(ids=ids_attempt, x='id_', y='correlation', save_name=pre + 'corr' + post)
        ufig.plot_multi_boxplots(ids=ids_attempt, x='id_', y='mae', save_name=pre + 'mae' + post)
        ufig.plot_multi_boxplots(ids=ids_attempt, x='id_', y='mse', save_name=pre + 'mse' + post)

    ufig.mean_weights(ids=ids_all, save_name=pre)
