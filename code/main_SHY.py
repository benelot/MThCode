import utilities_train as utrain
import utilities_figures as ufig

parameter = [['ID07_32h07m_0', 'general', 'ID07', [32, 7], 8],
             ['ID07_35h15m_0', 'general', 'ID07', [35, 15], 8],
             ['ID07_38h22m_0', 'general', 'ID07', [38, 22], 8],
             ['ID08_57h58m_0', 'general', 'ID08', [57, 58], 6],
             ['ID08_60h10m_0', 'general', 'ID08', [60, 10], 6],
             ['ID08_64h40m_0', 'general', 'ID08', [64, 40], 6],
             ['ID11_60h05m_0', 'general', 'ID11', [60, 5], 3],
             ['ID11_62h12m_0', 'general', 'ID11', [62, 12], 3],
             ['ID11_65h00m_0', 'general', 'ID11', [65, 0], 3],
             ['ID11_129h48m_0', 'general', 'ID11', [129, 48], 3],
             ['ID11_132h35m_0', 'general', 'ID11', [132, 35], 3],
             ['ID11_136h35m_0', 'general', 'ID11', [136, 35], 3]]


ids = []
for i, val in enumerate(parameter):
    ids.append(val[0])

    params = {'id_': ids[-1],
              'model_type': val[1],
              'path2data': '../data/',
              'patient_id': val[2],
              'time_begin': val[3],  # [hour, minute]
              'duration': 60,  # seconds
              # model parameters ------------------------
              'visible_size': 'all',  # 'all' or scalar
              'hidden_size': val[4],  # improve: portion
              'lambda': 0,
              'af': 'linear',  # 'relu', 'linear', 'sigmoid'
              'bias': True,
              'window_size': 30,
              # train parameters -------------------------
              'loss_function': 'mae',  # 'mse' or 'mae'
              'lr': 0.0007,
              'normalization': None,  # 'min_max', 'standard', None
              'epochs': 25}

    utrain.train_test(params, train_set=False)
    ufig.plot_train_test(ids[-1], [3, 12, 20, 26])

ufig.plot_multi_boxplots(ids=ids, x='id_', y='correlation', save_name='shy_0')
ufig.plot_multi_boxplots(ids=ids, x='id_', y='mae', save_name='shy_0')
ufig.plot_multi_boxplots(ids=ids, x='id_', y='mse', save_name='shy_0')

ufig.mean_weights(ids=ids, save_name='shy_0')
