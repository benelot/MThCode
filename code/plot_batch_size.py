import utilities_train as utrain
import utilities_figures as ufig
import utilities_various as uvar
import os

if __name__ == '__main__':

    batch_sizes = [2, 5, 50, 512]
    ids = []
    custom_set = False
    for _, val in enumerate(batch_sizes):
        for i in range(3):
            ids.append('batch_size_' + str(val) + '_allnorm_ID07_32h07m_' + str(i))
            if custom_set:
                custom_test_set = {'time_begin': [32, 7, 29.8],
                                   'duration': 0.2,
                                   'batch_size': 5}
                utrain.predict(ids[-1], custom_test_set=custom_test_set)
                utrain.distance(ids[-1])

            ids.append('batch_size_' + str(val) + '_allnorm_ID07_35h15m_' + str(i))
            if custom_set:
                custom_test_set = {'time_begin': [35, 15, 29.8],
                                   'duration': 0.2,
                                   'batch_size': 5}
                utrain.predict(ids[-1], custom_test_set=custom_test_set)
                utrain.distance(ids[-1])

            ids.append('batch_size_' + str(val) + '_allnorm_ID07_38h22m_' + str(i))
            if custom_set:
                custom_test_set = {'time_begin': [38, 22, 29.8],
                                   'duration': 0.2,
                                   'batch_size': 5}
                utrain.predict(ids[-1], custom_test_set=custom_test_set)
                utrain.distance(ids[-1])
        print('---------------------------------------------')

    ufig.plot_multi_boxplots(ids=ids, x='batch_size', y='correlation', hue='brain_state',
                             save_name='batch_size_allnorm', ylim=(0, 1))
    ufig.mean_weights(ids=ids, save_name='batch_size_allnorm', hidden=False, diagonal=True)
