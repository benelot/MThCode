import utilities_train as utrain
import utilities_figures as ufig
import utilities_various as uvar

ids = []
for j in range(8):
    pre = 'relu_'
    post = '_' + str(j)

    parameter = [[pre + 'ID07_32h07m' + post, 'general', 'ID07', [32, 7], 8, 0.0001],
                 [pre + 'ID07_35h15m' + post, 'general', 'ID07', [35, 15], 8, 0.0001],
                 [pre + 'ID07_38h22m' + post, 'general', 'ID07', [38, 22], 8, 0.0001],
                 [pre + 'ID08_57h58m' + post, 'general', 'ID08', [57, 58], 6, 0.0001],
                 [pre + 'ID08_60h10m' + post, 'general', 'ID08', [60, 10], 6, 0.0001],
                 [pre + 'ID08_64h40m' + post, 'general', 'ID08', [64, 40], 6, 0.0001],
                 [pre + 'ID11a_60h05m' + post, 'general', 'ID11', [60, 5], 3, 0.0001],
                 [pre + 'ID11a_62h12m' + post, 'general', 'ID11', [62, 12], 3, 0.0001],
                 [pre + 'ID11a_65h00m' + post, 'general', 'ID11', [65, 0], 3, 0.0001],
                 [pre + 'ID11b_129h48m' + post, 'general', 'ID11', [129, 48], 3, 0.0001],
                 [pre + 'ID11b_132h35m' + post, 'general', 'ID11', [132, 35], 3, 0.0001],
                 [pre + 'ID11b_136h35m' + post, 'general', 'ID11', [136, 35], 3, 0.0001]]

    for i in range(len(parameter)):
        ids.append(parameter[i][0])

        # params = {'id_': ids[-1],
        #           'model_type': parameter[i][1],
        #           'path2data': '../data/',
        #           'patient_id': parameter[i][2],
        #           'time_begin': parameter[i][3],  # [hour, minute]
        #           'duration': 60,  # seconds
        #           # model parameters ------------------------
        #           'visible_size': 'all',  # 'all' or scalar
        #           'hidden_size': parameter[i][4],  # improve: portion
        #           'lambda': 0,
        #           'af': 'relu',  # 'relu', 'linear', 'sigmoid'
        #           'bias': False,
        #           'window_size': 30,
        #           # train parameters -------------------------
        #           'loss_function': 'mae',  # 'mse' or 'mae'
        #           'lr': parameter[i][5],
        #           'normalization': 'min_max_positive',  # 'min_max', 'standard', None
        #           'epochs': 25}

        # utrain.train_test(params, train_set=False)
        # utrain.test_train_set(ids[-1])
        # ufig.plot_train_test(ids[-1], [3, 12, 23, 29], lim_nr_samples=2000, predict_train_set=True)
        # print(str(i))

    #print('--------------- Attempt: ' + str(j))
    #ufig.plot_multi_boxplots(ids=ids, x='id_', y='correlation', hue='train_set', save_name='relu_shy' + post + '_corr')
    # ufig.plot_multi_boxplots(ids=ids, x='id_', y='mae', save_name='relu_shy' + post + '_mae')
    # ufig.plot_multi_boxplots(ids=ids, x='id_', y='mse', save_name='relu_shy' + post + '_mse')

ufig.mean_weights(ids=ids, hidden=False, save_name='relu_shy_all')
