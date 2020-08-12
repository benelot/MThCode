import utilities_train as utrain
import utilities_figures as ufig
import utilities_various as uvar
import os

if __name__ == '__main__':

    batch_sizes = [1, 2, 5, 50, 512]
    ids = []
    pre = '_wd'
    custom_set = False
    for _, val in enumerate(batch_sizes):
        for i in range(3):
            ids.append('batch_size_' + str(val) + pre + '_ID07_32h07m_' + str(i))
            if custom_set:
                custom_test_set = {'time_begin': [32, 7, 20],
                                   'duration': 10,
                                   'batch_size': 50}
                utrain.predict(ids[-1], custom_test_set=custom_test_set)
                utrain.distance(ids[-1])
            print('0')
            ids.append('batch_size_' + str(val) + pre + '_ID07_35h15m_' + str(i))
            if custom_set:
                custom_test_set = {'time_begin': [35, 15, 20],
                                   'duration': 10,
                                   'batch_size': 50}
                utrain.predict(ids[-1], custom_test_set=custom_test_set)
                utrain.distance(ids[-1])
            print('1')
            ids.append('batch_size_' + str(val) + pre + '_ID07_38h22m_' + str(i))
            if custom_set:
                custom_test_set = {'time_begin': [38, 22, 20],
                                   'duration': 10,
                                   'batch_size': 50}
                utrain.predict(ids[-1], custom_test_set=custom_test_set)
                utrain.distance(ids[-1])
            print('2')
        print('---------------------------------------------')

    ufig.plot_multi_boxplots(ids=ids, x='batch_size', y='correlation', hue='brain_state',
                             save_name='batch_size' + pre)
    ufig.plot_multi_boxplots(ids=ids, x='batch_size', y='mae', hue='brain_state',
                             save_name='batch_size_mae_' + pre)
    ufig.mean_weights(ids=ids, save_name='batch_size' + pre, hidden=False, diagonal=True)
