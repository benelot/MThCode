import utilities_train as utrain
import utilities_figures as ufig

parameter = [['ID07_parallel', 'parallel', 'ID07', [32, 7], 8, 500],
             ['ID07_sequential', 'general', 'ID07', [32, 7], 8, 30],
             ['ID08_parallel', 'parallel', 'ID08', [57, 58], 6, 500],
             ['ID08_sequential', 'general', 'ID08', [57, 58], 6, 30],
             ['ID11_parallel', 'parallel', 'ID11', [60, 5], 3, 500],
             ['ID11_sequential', 'general', 'ID11', [60, 5], 3, 30]]

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
              'lr': 0.0005,
              'normalization': 'standard',  # 'min_max', 'standard', None
              'epochs': val[5]}

    utrain.train_test(params, train_set=False)
    ufig.plot_train_test(ids[-1], [3, 12, 20, 26])

ufig.plot_multi_boxplots(ids=ids, x='id_', y='correlation', save_name='par_seq_corr')
ufig.plot_multi_boxplots(ids=ids, x='id_', y='mae', save_name='par_seq_mae')
ufig.plot_multi_boxplots(ids=ids, x='id_', y='mse', save_name='par_seq_mse')
