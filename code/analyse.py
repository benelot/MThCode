import utilities_train as utrain
import utilities_figures as ufig
import utilities_various as uvar
import pickle
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

if __name__ == '__main__':

    # params = {'id_': 'small_bs50',
    #           'model_type': None,  # To be removed
    #           'path2data': '../data/',
    #           'patient_id': 'ID07',
    #           'time_begin': [32, 7],  # [hour, minute]
    #           'duration': 30,  # seconds
    #           'brain_state': 'beginning',
    #           'add_id': '(M)',
    #           # model parameters ------------------------
    #           'visible_size': 10,  # 'all' or scalar
    #           'hidden_size': 0,  # improve: portion
    #           'lambda': 0,
    #           'af': 'relu',  # 'relu', 'linear', 'sigmoid'
    #           'bias': True,
    #           'window_size': 30,
    #           # train parameters -------------------------
    #           'loss_function': 'mae',  # 'mse' or 'mae'
    #           'lr': 0.001,
    #           'batch_size': 50,
    #           'shuffle': False,
    #           'normalization': 'standard_positive',  # 'min_max', 'standard', None
    #           'epochs': 30}
    #
    # utrain.train_and_test(params)
    # ufig.plot_train_test('small_bs50', [2, 4, 6, 8], lim_nr_samples=2000)

    id0, id1 = 'small', 'small_bs50'
    W_01 = pickle.load(open('../models/' + id0 + '/W_epoch.pkl', 'rb'))
    W_01 = np.asarray(W_01['W_epoch'])
    W_50 = pickle.load(open('../models/' + id1 + '/W_epoch.pkl', 'rb'))
    W_50 = np.asarray(W_50['W_epoch'])

    sns.set_style('whitegrid')
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 4)

    mask = np.zeros((W_50.shape[1], W_50.shape[1]))
    ax0 = fig.add_subplot(gs[:1, 2:])
    for i in range(W_50.shape[1]):
        plt.plot(W_50[1:, i, i], color='blue')
        mask[i, i] = -1
        if i != 9:
            plt.plot(W_50[1:, i, i + 1], color='red')
            mask[i, i + 1] = 1
        if i != 0:
            plt.plot(W_50[1:, i, i - 1], color='red')
            mask[i, i - 1] = 1
    ax0.set_xlabel('Finished epoch'), ax0.set_ylabel('Weight'), ax0.set_title('Weight progression with BS = 50')
    ax0.set_xlim(left=0)

    ax1 = fig.add_subplot(gs[:, :1])
    sns.heatmap(mask, cmap='bwr', cbar=False, square=True)
    ax1.set_xlabel('From node'), ax1.set_ylabel('To node'), ax1.set_title('Weight index color code')

    ax2 = fig.add_subplot(gs[1:, 2:])
    for i in range(W_01.shape[1]):
        plt.plot(W_01[:, i, i], color='blue')
        if i != 9:
            plt.plot(W_01[:, i, i + 1], color='red')
        if i != 0:
            plt.plot(W_01[:, i, i - 1], color='red')
    ax2.set_xlabel('Finished epoch'), ax2.set_ylabel('Weight'), ax2.set_title('Weight progression with BS = 1')
    ax2.set_xlim(left=0)

    ax3 = fig.add_subplot(gs[:1, 1:2])
    sns.heatmap(W_50[-1, :, :], cmap='bwr', vmin=-1, vmax=1, cbar=False, square=True)
    ax3.set_xlabel('From node'), ax3.set_ylabel('To node'), ax3.set_title('Weight matrix of last epoch with BS = 50')

    ax4 = fig.add_subplot(gs[1:, 1:2])
    sns.heatmap(W_01[-1, :, :], cmap='bwr', vmin=-1, vmax=1, cbar=False, square=True)
    ax4.set_xlabel('From node'), ax4.set_ylabel('To node'), ax4.set_title('Weight matrix of last epoch with BS = 1')

    plt.tight_layout()
    plt.show()
    test = 1



