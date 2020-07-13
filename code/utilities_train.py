""" Various functions

Part of master thesis Segessenmann J. (2020)
"""

import numpy as np
from scipy.io import loadmat
from scipy.stats import mode
from scipy import signal
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import time
import torch
import torch.nn as nn
import pickle
import os

import models


def train_test(params: dict, train_set=False):
    train(params)
    predict(params['id_'], train_set=train_set)
    distance(params['id_'], train_set=False)
    if train_set:
        distance(params['id_'], train_set=True)


def data_loader(id_: str=None, params: dict=None, train_portion=0.8, windowing=True):
    """ Loads and prepares iEEG data.

        Returns:
            X_train, X_test
        Or (if windowing=False):
            train_set, test_set
    """
    # Load and cut data
    if params is None:
        params = pickle.load(open('../models/' + id_ + '/params.pkl', 'rb'))
    data_mat = loadmat(params['path2data'] + params['patient_id'] + '_' + str(params['time_begin'][0]) + 'h.mat')
    info_mat = loadmat(params['path2data'] + params['patient_id'] + '_' + 'info.mat')
    fs = int(info_mat['fs'])
    sample_begin = int(params['time_begin'][1] * 60 * fs)
    sample_end = int(sample_begin + params['duration'] * fs)
    if params['visible_size'] != 'all':
        data = data_mat['EEG'][:params['visible_size'], sample_begin:sample_end].transpose()
    else:
        data = data_mat['EEG'][:, sample_begin:sample_end].transpose()

    # Resample to fs of 512 Hz
    if fs != 512:
        data = signal.resample(data, num=int(data.shape[0] / fs * 512), axis=0)

    # Normalization
    if params['normalization'] == 'standard_positive':
        data = (data - np.mean(data, axis=0)) / (5 * np.std(data, axis=0))
        data = data / 2 + 0.5
    elif params['normalization'] == 'standard':
        sc = StandardScaler()
        data = sc.fit_transform(data) / 5
    elif params['normalization'] == 'min_max':
        sc = MinMaxScaler(feature_range=(-1, 1))
        data = sc.fit_transform(data)
    elif params['normalization'] == 'min_max_positive':
        sc = MinMaxScaler(feature_range=(0, 1))
        data = sc.fit_transform(data)
    elif params['normalization'] is not None:
        print('Error: No valid normalization method.')

    # To tensor
    data = torch.FloatTensor(data)

    # Split data into training and test set
    train_set = data[:int(train_portion * data.shape[0]), :]
    test_set = data[int(train_portion * data.shape[0]):, :]

    if windowing is False:
        return train_set, test_set

    # Windowing
    X_train, X_test = [], []
    for i in range(train_set.shape[0] - params['window_size']):
        X_train.append(train_set[i:i + params['window_size'], :])
    for i in range(test_set.shape[0] - params['window_size']):
        X_test.append(test_set[i:i + params['window_size'], :])

    return X_train, X_test


def train(params: dict):
    """ Trains model with parameters params.

        Saves:
            model.pth
            params.pkl
            eval_optim.pkl
    """
    # Load data
    X_train, X_test = data_loader(params=params)
    if params['visible_size'] == 'all':
        params['visible_size'] = X_train[0].shape[1]

    # Define model, criterion and optimizer
    model = None
    if params['model_type'] == 'general':
        model = models.general_RNN(params)
    elif params['model_type'] == 'single':
        get_best_node(params, X_train)
        return None
    elif params['model_type'] == 'parallel':
        train_parallel(params, X_train)
        return None
    else:
        print('Error: No valid model type.')

    # Training parameters
    criterion = None
    if params['loss_function'] == 'mae':
        criterion = nn.L1Loss(reduction='none')
    elif params['loss_function'] == 'mse':
        criterion = nn.MSELoss(reduction='none')
    else:
        print('Error: No valid loss function.')
    lr = params['lr']
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training
    temp_loss = np.zeros([len(X_train), model.visible_size])
    epoch_loss = np.zeros([params['epochs'], model.visible_size])
    epoch_grad_norm = np.zeros(params['epochs'])
    temp_grad_norm = np.zeros(len(X_train))

    start_time = time.time()

    for epoch in range(params['epochs']):
        for T, X in enumerate(X_train):
            optimizer.zero_grad()
            prediction = model(X)
            loss = criterion(prediction, X[-1, :])
            temp_loss[T, :] = loss.detach()
            torch.autograd.backward(loss.mean())
            optimizer.step()
            for p in model.parameters():
                temp_grad_norm[T] = p.grad.data.norm(2).item()
        epoch_grad_norm[epoch] = np.mean(temp_grad_norm)
        epoch_loss[epoch, :] = np.mean(temp_loss, axis=0)
        print(f'Epoch: {epoch} | Loss: {np.mean(temp_loss):.4}')

    total_time = time.time() - start_time
    print(f'Time [min]: {total_time / 60:.3}')

    # Make optimizer evaluation dictionary
    eval_optimization = {'id_': params['id_'],
                         'loss': epoch_loss,
                         'grad_norm': epoch_grad_norm}

    # Save model
    directory = '../models/' + params['id_']
    if not os.path.exists(directory):
        os.mkdir(directory)
    torch.save(model.state_dict(), directory + '/model.pth')
    pickle.dump(params, open(directory + '/params.pkl', 'wb'))
    pickle.dump(eval_optimization, open(directory + '/eval_optimization.pkl', 'wb'))


def predict(id_: str, train_set=False):
    """ Tests model an returns and saves predicted values.

        Returns:
            eval_prediction

        Saves:
            eval_prediction.pkl
    """
    # Load parameters
    params = pickle.load(open('../models/' + id_ + '/params.pkl', 'rb'))

    # Load data
    X_train, X_test = data_loader(params=params)

    # Get trained model
    model = None
    if params['model_type'] == 'general':
        model = models.general_RNN(params)
    elif params['model_type'] == 'single':
        if train_set:
            predict_by_best_node(params['id_'], X_test, X_train=X_train)
        else:
            predict_by_best_node(params['id_'], X_test)
        return None
    elif params['model_type'] == 'parallel':
        if train_set:
            predict_parallel(params['id_'], X_test, X_train=X_train)
        else:
            predict_parallel(params['id_'], X_test)
        return None
    else:
        print('Error: No valid model type.')

    model.load_state_dict(torch.load('../models/' + id_ + '/model.pth'))

    # Evaluate model
    model.eval()

    with torch.no_grad():
        test_pred = np.zeros((len(X_test), model.visible_size))
        test_true = np.zeros((len(X_test), model.visible_size))
        for T, X in enumerate(X_test):
            predictions = model(X)
            test_pred[T, :] = predictions.numpy()
            test_true[T, :] = X[-1, :].numpy()

        eval_prediction = {'id_': id_,
                           'test_pred': test_pred,
                           'test_true': test_true}
        if train_set is False:
            pickle.dump(eval_prediction, open('../models/' + id_ + '/eval_prediction.pkl', 'wb'))
            return eval_prediction
        else:
            train_pred = np.zeros((len(X_train), model.visible_size))
            train_true = np.zeros((len(X_train), model.visible_size))
            with torch.no_grad():
                for T, X in enumerate(X_train):
                    predictions = model(X)
                    train_pred[T, :] = predictions.numpy()
                    train_true[T, :] = X[-1, :].numpy()

    eval_prediction['train_pred'] = train_pred
    eval_prediction['train_true'] = train_true
    pickle.dump(eval_prediction, open('../models/' + id_ + '/eval_prediction.pkl', 'wb'))

    return eval_prediction


def distance(id_: str, train_set=False):
    """ Computes Correlation, MSE, MAE for evaluation.

        Returns:
            eval_distances

        Saves:
            eval_distances.pkl
    """
    # Load parameters
    params = pickle.load(open('../models/' + id_ + '/params.pkl', 'rb'))
    eval_prediction = pickle.load(open('../models/' + id_ + '/eval_prediction.pkl', 'rb'))

    node_size = params['visible_size']

    if train_set:
        pred = eval_prediction['train_pred']
        true = eval_prediction['train_true']
        train_str = '_train'
    else:
        pred = eval_prediction['test_pred']
        true = eval_prediction['test_true']
        train_str = ''

    # Calculate distances
    corr = []
    for i in range(node_size):
        corr.append(np.corrcoef(pred[:, i], true[:, i])[0, 1])
    mse = np.mean((pred - true) ** 2, axis=0)
    mae = np.mean(np.abs(pred - true), axis=0)

    eval_distances = {'id_': [id_ for _ in range(node_size)],
                      'node_idx': [i for i in range(node_size)],
                      'train_set': [False for _ in range(node_size)],
                      'channel_size': [node_size for _ in range(node_size)],
                      'hidden_size': [params['hidden_size'] for i in range(node_size)],
                      'af': [params['af'] for _ in range(node_size)],
                      'bias': [params['bias'] for _ in range(node_size)],
                      'lambda': [params['lambda'] for _ in range(node_size)],
                      'correlation': corr,
                      'mse': mse,
                      'mae': mae}

    if train_set is True:
        eval_distances['train_set'] = [True for _ in range(node_size)]

    pickle.dump(eval_distances, open('../models/' + id_ + '/eval_distances' + train_str + '.pkl', 'wb'))

    return eval_distances


def get_best_node(params, X_train):
    inv_diagmat = np.abs(np.identity(X_train[0].shape[1]) - 1)
    best_nodes = np.zeros((len(X_train), X_train[0].shape[1]))
    for T, X in enumerate(X_train):
        corrmat = np.multiply(np.corrcoef(X.T), inv_diagmat)
        best_nodes[T, :] = np.argmax(np.abs(corrmat), axis=0)
    # most frequent node
    best_node = (mode(best_nodes, axis=0)[0][0]).astype(int)
    # Save best node
    directory = '../models/' + params['id_']
    if not os.path.exists(directory):
        os.mkdir(directory)
    np.save(directory + '/best_node.npy', best_node)
    pickle.dump(params, open(directory + '/params.pkl', 'wb'))


def predict_by_best_node(id_, X_test, X_train=None):
    best_node = np.load('../models/' + id_ + '/best_node.npy')
    test_pred = np.zeros((len(X_test), best_node.shape[0]))
    test_true = np.zeros((len(X_test), best_node.shape[0]))
    for T, X in enumerate(X_test):
        test_pred[T, :] = X[-1, best_node.tolist()].numpy()
        test_true[T, :] = X[-1, :].numpy()
    eval_prediction = {'id_': id_,
                       'test_pred': test_pred,
                       'test_true': test_true}
    if X_train is None:
        pickle.dump(eval_prediction, open('../models/' + id_ + '/eval_prediction.pkl', 'wb'))
        return eval_prediction
    else:
        print('Error: Has yet to be implemented.')
        return eval_prediction


def train_parallel(params, X_train):
    # Make model
    model = models.parallel_RNN(params)

    # Pack data to matrices (currently not used)
    # n_packs = int(len(X_train)/512)  # One pack per second
    # n_wins_per_pack = int(len(X_train)/n_packs)
    # X_train_mats = []
    # for i in range(n_packs):
    #     win_begin = i * n_wins_per_pack
    #     win_end = win_begin + n_wins_per_pack
    #     X_train_mats.append(np.concatenate(X_train[win_begin:win_end], axis=1))
    # if win_end < len(X_train):
    #     X_train_mats.append(np.concatenate(X_train[win_end:], axis=1))

    # Pack data to single matrix for late training
    X_train_big_mat = np.concatenate(X_train, axis=1)

    # Training parameters
    criterion = None
    if params['loss_function'] == 'mae':
        criterion = nn.L1Loss(reduction='none')
    elif params['loss_function'] == 'mse':
        criterion = nn.MSELoss(reduction='none')
    else:
        print('Error: No valid loss function.')

    lr = params['lr']
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training
    epoch_loss = np.zeros([params['epochs'], model.visible_size])
    epoch_grad_norm = np.zeros(params['epochs'])

    start_time = time.time()

    for epoch in range(params['epochs']):
        optimizer.zero_grad()
        prediction = model(X_train_big_mat)
        loss = criterion(prediction, torch.from_numpy(X_train_big_mat[-1, :]))
        torch.autograd.backward(loss.mean())
        optimizer.step()
        for p in model.parameters():
            epoch_grad_norm[epoch] = p.grad.data.norm(2).item()
        epoch_loss[epoch, :] = np.mean(loss.detach().numpy().reshape((-1, model.visible_size)), axis=0)
        # if epoch % 20 == 0:
        print(f'Epoch: {epoch} | Loss: {np.mean(epoch_loss[epoch, :]):.4}')

    total_time = time.time() - start_time
    print(f'Time [min]: {total_time / 60:.3}')

    # Make optimizer evaluation dictionary
    eval_optimization = {'id_': params['id_'],
                         'loss': epoch_loss,
                         'grad_norm': epoch_grad_norm}

    # Save model
    directory = '../models/' + params['id_']
    if not os.path.exists(directory):
        os.mkdir(directory)
    torch.save(model.state_dict(), directory + '/model.pth')
    pickle.dump(params, open(directory + '/params.pkl', 'wb'))
    pickle.dump(eval_optimization, open(directory + '/eval_optimization.pkl', 'wb'))


def predict_parallel(id_, X_test, X_train=None):
    # Load parameters
    params = pickle.load(open('../models/' + id_ + '/params.pkl', 'rb'))

    # Load data
    X_train, X_test = data_loader(params=params)

    # Prepare data
    X_test_mat = np.concatenate(X_test, axis=1)

    model = models.parallel_RNN(params)
    model.load_state_dict(torch.load('../models/' + id_ + '/model.pth'))

    # Evaluate model
    model.eval()

    with torch.no_grad():
        predictions = model(X_test_mat)
        test_pred = predictions.numpy().reshape((len(X_test), model.visible_size))
        test_true = X_test_mat[-1, :].reshape((len(X_test), model.visible_size))

        eval_prediction = {'id_': id_,
                           'test_pred': test_pred,
                           'test_true': test_true}
        if X_train is None:
            pickle.dump(eval_prediction, open('../models/' + id_ + '/eval_prediction.pkl', 'wb'))
            return eval_prediction
        else:
            X_train_mat = np.concatenate(X_train, axis=1)
            predictions = model(X_train_mat)
            train_pred = predictions.numpy().reshape((len(X_train), model.visible_size))
            train_true = X_train_mat[-1, :].reshape((len(X_train), model.visible_size))

    eval_prediction['train_pred'] = train_pred
    eval_prediction['train_true'] = train_true
    pickle.dump(eval_prediction, open('../models/' + id_ + '/eval_prediction.pkl', 'wb'))

    return eval_prediction
