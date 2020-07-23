import utilities_train as utrain
import utilities_figures as ufig

if __name__ == '__main__':

    for attempt in range(3):
        print('------------------------------ ' + 'Attempt Nr. ' + str(attempt) + ' ------------------------------')
        pre = 'linear_'
        post = '_' + str(attempt)

        parameter = [[pre + 'ID07_32h07m' + post, 'ID07', [32, 7]],
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

        ids = []
        for i, val in enumerate(parameter):
            ids.append(val[0])

            params = {'id_': ids[-1],
                      'model_type': None,  # To be removed
                      'path2data': '../data/',
                      'patient_id': val[1],
                      'time_begin': val[2],  # [hour, minute]
                      'duration': 5,  # seconds
                      # model parameters ------------------------
                      'visible_size': 30,  # 'all' or scalar
                      'hidden_size': 0,  # improve: portion
                      'lambda': 0,
                      'af': 'linear',  # 'relu', 'linear', 'sigmoid'
                      'bias': True,
                      'window_size': 30,
                      # train parameters -------------------------
                      'loss_function': 'mae',  # 'mse' or 'mae'
                      'lr': 0.0002,
                      'normalization': 'standard',  # 'min_max', 'standard', None
                      'epochs': 400}

            utrain.train_and_test(params)
            utrain.test_train_set(ids[-1])
            ufig.plot_train_test(ids[-1], [3, 8, 13, 17], lim_nr_samples=2000, predict_train_set=True)

        ufig.plot_multi_boxplots(ids=ids, x='id_', y='correlation', hue='train_set', save_name=pre + 'corr' + post)
        ufig.plot_multi_boxplots(ids=ids, x='id_', y='mae', hue='train_set', save_name=pre + 'mae' + post)
        ufig.plot_multi_boxplots(ids=ids, x='id_', y='mse', hue='train_set', save_name=pre + 'mse' + post)

        ufig.mean_weights(ids=ids, save_name='linear' + post)
