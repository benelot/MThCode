import utilities_train as utrain
import utilities_figures as ufig
import utilities_various as uvar

params = {'id_': 'artificial_wd',
          'model_type': 'single_layer',  # To be removed
          'path2data': '../data/',
          'patient_id': 'ID07',
          'time_begin': [20, 1],  # [hour, minute]
          'artificial_signal': None,  # [bool AS, bool small_weights]
          'duration': 30,  # seconds
          'brain_state': '',
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
          'lr': 0.001,
          'batch_size': 512,
          'shuffle': True,
          'normalization': 'all_standard_positive',  # 'min_max', 'standard', None
          'epochs': 250}

params0 = params.copy()
params1 = params.copy()

params0['id_'] = params['id_'] + '_strong_coupling'
params0['brain_state'] = 'strong_coupling'
params0['artificial_signal'] = [True, False]

params1['id_'] = params['id_'] + '_weak_coupling'
params1['brain_state'] = 'weak_coupling'
params1['artificial_signal'] = [True, True]

utrain.train_and_test(params0)
ufig.plot_train_test(params0['id_'], n_nodes=6)

utrain.train_and_test(params1)
ufig.plot_train_test(params1['id_'], n_nodes=6)

ids = [params0['id_'], params1['id_']]

ufig.plot_multi_boxplots(ids=ids, x='patient_id', y='mae', hue='brain_state', save_name=params['id_'])
ufig.mean_weights(ids=ids, save_name=params['id_'])
