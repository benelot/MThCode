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
import matplotlib.pyplot as plt
import random
import seaborn as sns
import cmath

import models


def train_and_test(params: dict):
    if params['model_type'] == 'general' or params['model_type'] is None:
        train(params)
        predict(params['id_'])
        distance(params['id_'])
    elif params['model_type'] == 'single_layer':
        train_single_layer(params)
        predict_single_layer(params['id_'])
        distance(params['id_'])
    else:
        print('Error: No valid model type.')


def pre_process(id_: str=None, params: dict=None, custom_test_set=None, artificial=False):
    """ Loads and prepares iEEG data.

        Returns:
            train_set, test_set
        Or (if windowing=True):
            X_train, X_test
    """
    # Load and cut data
    if params is None:
        params = pickle.load(open('../models/' + id_ + '/params.pkl', 'rb'))
    if custom_test_set is not None:
        params['time_begin'] = custom_test_set['time_begin']
        params['duration'] = custom_test_set['duration']

    if not artificial:
        data_mat = loadmat(params['path2data'] + params['patient_id'] + '_' + str(params['time_begin'][0]) + 'h.mat')
        info_mat = loadmat(params['path2data'] + params['patient_id'] + '_' + 'info.mat')
        fs = float(info_mat['fs'])
        if len(params['time_begin']) == 3:
            sample_begin = int(np.round(params['time_begin'][1] * 60 * fs + params['time_begin'][2] * fs))
        else:
            sample_begin = int(np.round(params['time_begin'][1] * 60 * fs))
        sample_end = int(np.round(sample_begin + params['duration'] * fs))
        if params['visible_size'] != 'all':
            data = data_mat['EEG'][:params['visible_size'], sample_begin:sample_end].transpose()
        else:
            data = data_mat['EEG'][:, sample_begin:sample_end].transpose()

        # Resample
        if params['resample'] != fs and params['resample'] is not None:
            data = signal.resample(data, num=int(data.shape[0] / fs * params['resample']), axis=0)

    else:
        data = coupled_oscillator(t_length=params['duration'], fs=params['resample'],
                                  small_weights=params['artificial_signal'][1])

    # Normalize
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
    elif params['normalization'] == 'all_standard_positive':
        data = (data - np.mean(data)) / (3 * np.std(data))  # 5
        data = data / 2 + 0.5
    elif params['normalization'] == 'all_standard':
        data = (data - np.mean(data)) / np.std(data)
    elif params['normalization'] is not None:
        print('Error: No valid normalization method.')

    # To tensor
    data = torch.from_numpy(data)

    return data


def train(params):
    """ Trains model with parameters params.

        Saves:
            ../model/model.pth
            ../model/params.pkl
            ../model/eval_optim.pkl
    """
    # Define device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data
    print('Status: Start data preparation.')
    data_pre = pre_process(params=params)
    if params['visible_size'] == 'all':
        params['visible_size'] = data_pre.shape[1]
    data_set = iEEG_DataSet(data_pre, params['window_size'])
    data_generator = torch.utils.data.DataLoader(data_set, batch_size=params['batch_size'], shuffle=params['shuffle'])

    # Make model
    model = models.GeneralRNN(params)
    model = model.to(device)

    # Define training parameters
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
    loss = None
    epoch_loss = np.zeros([params['epochs'], model.visible_size])
    epoch_grad_norm = np.zeros(params['epochs'])
    #epoch_time = np.zeros(params['epochs'] + 1)
    W = []

    start_time = time.time()
    #epoch_time[0] = time.time()
    print('Status: Start training with cuda = ' + str(torch.cuda.is_available()) + '.')

    for epoch in range(params['epochs']):
        W.append(np.copy(model.W.weight.data.cpu().numpy()))
        for X, y in data_generator:
            X, y = X.to(device), y.float().to(device)
            optimizer.zero_grad()
            prediction = model(X)
            loss = criterion(prediction, y)
            torch.autograd.backward(loss.mean()) #  loss.mean().backward()
            optimizer.step()
        for p in model.parameters():
            epoch_grad_norm[epoch] = p.grad.data.norm(2).item()
        epoch_loss[epoch, :] = np.mean(loss.detach().cpu().numpy(), axis=0)
        #epoch_time[epoch + 1] = time.time() - epoch_time[epoch]
        #if epoch % 5 == 0:

        add_id = params['add_id']
        print(f'{add_id} Epoch: {epoch} | Loss: {np.mean(epoch_loss[epoch, :]):.4}')

    total_time = time.time() - start_time
    print(f'Time [min]: {total_time / 60:.3}')

    # Make optimizer evaluation dictionary
    eval_optimization = {'id_': params['id_'],
                         'loss': epoch_loss,
                         'grad_norm': epoch_grad_norm}

    # Save model
    print('Status: Save trained model to file.')
    directory = '../models/' + params['id_']
    if not os.path.exists(directory):
        os.mkdir(directory)
    torch.save(model.state_dict(), directory + '/model.pth')
    pickle.dump(params, open(directory + '/params.pkl', 'wb'))
    pickle.dump(eval_optimization, open(directory + '/eval_optimization.pkl', 'wb'))

    # W_epoch = {'W_epoch': W}
    # pickle.dump(W_epoch, open(directory + '/W_epoch.pkl', 'wb'))


def predict(id_: str, custom_test_set: dict=None):
    """ Tests model an returns and saves predicted values.

        If the prediction set is not the training set, pass a custom_test_set dictionary containing:
            'time_begin', 'duration', 'batch_size'

        Returns and saves:
            ../model/eval_prediction.pkl
    """
    # Define device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data and parameters
    print('Status: Load and process data for prediction.')
    params = pickle.load(open('../models/' + id_ + '/params.pkl', 'rb'))
    if custom_test_set is None:
        data_pre = pre_process(params=params)
        batch_size = params['batch_size']
    else:
        data_pre = pre_process(params=params, custom_test_set=custom_test_set)
        batch_size = custom_test_set['batch_size']
    data_set = iEEG_DataSet(data_pre, params['window_size'])
    data_generator = torch.utils.data.DataLoader(data_set, batch_size=batch_size, shuffle=False)

    # Make model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = models.GeneralRNN(params)
    model.load_state_dict(torch.load('../models/' + id_ + '/model.pth', map_location=device))
    model = model.to(device)

    # Evaluate model
    model.eval()
    pred_all = []
    true_all = []

    print('Status: Start prediction with cuda = ' + str(torch.cuda.is_available()) + '.')
    with torch.no_grad():
        for X, y in data_generator:
            X, y = X.to(device), y.to(device)
            predictions = model(X)
            pred_all.append(predictions.cpu().numpy())
            true_all.append(y.cpu().numpy())
        pred_all, true_all = np.concatenate(pred_all), np.concatenate(true_all)

    # Save predictions to file
    print('Status: Save predictions to file.')
    eval_prediction = {'prediction': pred_all,
                       'true': true_all}
    pickle.dump(eval_prediction, open('../models/' + id_ + '/eval_prediction.pkl', 'wb'))

    return eval_prediction


def distance(id_: str):
    """ Computes Correlation, MSE, MAE for evaluation.

        Returns and saves:
            ../model/eval_distances.pkl
    """
    # Load parameters
    params = pickle.load(open('../models/' + id_ + '/params.pkl', 'rb'))
    eval_prediction = pickle.load(open('../models/' + id_ + '/eval_prediction.pkl', 'rb'))

    node_size = params['visible_size']
    pred = eval_prediction['prediction']
    true = eval_prediction['true']

    # Calculate distances
    print('Status: Get distance metrics.')
    corr = []
    for i in range(node_size):
        corr.append(np.corrcoef(pred[:, i], true[:, i])[0, 1])
    mse = np.mean((pred - true) ** 2, axis=0)
    mae = np.mean(np.abs(pred - true), axis=0)

    eval_distances = {'id_': [id_ for _ in range(node_size)],
                      'patient_id': [params['patient_id'] for _ in range(node_size)],
                      'brain_state': [params['brain_state'] for _ in range(node_size)],
                      'node_idx': [i for i in range(node_size)],
                      'channel_size': [node_size for _ in range(node_size)],
                      'hidden_size': [params['hidden_size'] for i in range(node_size)],
                      'af': [params['af'] for _ in range(node_size)],
                      'bias': [params['bias'] for _ in range(node_size)],
                      'lambda': [params['lambda'] for _ in range(node_size)],
                      'batch_size': [params['batch_size'] for _ in range(node_size)],
                      'shuffle': [params['shuffle'] for _ in range(node_size)],
                      'normalization': [params['normalization'] for _ in range(node_size)],
                      'correlation': corr,
                      'mse': mse,
                      'mae': mae}

    pickle.dump(eval_distances, open('../models/' + id_ + '/eval_distances.pkl', 'wb'))

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


class iEEG_DataSet(torch.utils.data.Dataset):
    def __init__(self, data, window_size):
        self.data = data
        self.window_size = window_size

    def __getitem__(self, index):
        X = self.data[index:index + self.window_size, :]
        y = self.data[index + self.window_size, :]
        return X, y

    def __len__(self):
        return self.data.shape[0] - self.window_size


class iEEG_SingleLayerSet(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        x = self.data[index, :]
        y = self.data[index + 1, :]
        return x, y

    def __len__(self):
        return self.data.shape[0] - 1


def train_old(params: dict):
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


def predict_old(id_: str, train_set=False):
    """ Tests model an returns and saves predicted values.

        Returns:
            eval_prediction

        Saves:
            eval_prediction.pkl
    """
    # Load parameters
    params = pickle.load(open('../models/' + id_ + '/params.pkl', 'rb'))

    # Load data
    X_train, X_test = pre_process(params=params, windowing=True)

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


def train_single_layer(params):
    """ Trains model with parameters params.

        Saves:
            ../model/model.pth
            ../model/params.pkl
            ../model/eval_optim.pkl
    """
    # Define device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data
    print('Status: Start data preparation.')
    data_pre = pre_process(params=params, artificial=params['artificial_signal'][0])
    if params['visible_size'] == 'all':
        params['visible_size'] = data_pre.shape[1]
    data_set = iEEG_SingleLayerSet(data_pre)
    data_generator = torch.utils.data.DataLoader(data_set, batch_size=params['batch_size'], shuffle=params['shuffle'])

    # Make model
    model = models.SingleLayer(params)
    model = model.to(device)

    # Define training parameters
    criterion = None
    if params['loss_function'] == 'mae':
        criterion = nn.L1Loss(reduction='none')
    elif params['loss_function'] == 'mse':
        criterion = nn.MSELoss(reduction='none')
    else:
        print('Error: No valid loss function.')

    lr = params['lr']
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=params['weight_decay'])

    # Training
    loss = None
    epoch_loss = np.zeros([params['epochs'], model.visible_size])
    epoch_grad_norm = np.zeros(params['epochs'])
    W = []

    start_time = time.time()
    print('Status: Start training with cuda = ' + str(torch.cuda.is_available()) + '.')

    for epoch in range(params['epochs']):
        W.append(np.copy(model.W.weight.data.cpu().numpy()))
        for x, y in data_generator:
            x, y = x.float().to(device), y.float().to(device)
            optimizer.zero_grad()
            with torch.no_grad():
                model.W.weight.data = model.W.weight.data * \
                                      (torch.ones(1).to(device) - torch.eye(x.shape[1]).to(device))
            prediction = model(x)
            loss = criterion(prediction, y)
            torch.autograd.backward(loss.mean())
            optimizer.step()
        for p in model.parameters():
            epoch_grad_norm[epoch] = p.grad.data.norm(2).item()
        epoch_loss[epoch, :] = np.mean(loss.detach().cpu().numpy(), axis=0)
        if epoch % 10 == 0:
            add_id = params['add_id']
            print(f'{add_id} Epoch: {epoch} | Loss: {np.mean(epoch_loss[epoch, :]):.4}')

    with torch.no_grad():
        model.W.weight.data = model.W.weight.data * \
                              (torch.ones(1).to(device) - torch.eye(model.visible_size).to(device))
    total_time = time.time() - start_time
    print(f'Time [min]: {total_time / 60:.3}')

    # Make optimizer evaluation dictionary
    eval_optimization = {'id_': params['id_'],
                         'loss': epoch_loss,
                         'grad_norm': epoch_grad_norm}

    # Save model
    print('Status: Save trained model to file.')
    directory = '../models/' + params['id_']
    if not os.path.exists(directory):
        os.mkdir(directory)
    torch.save(model.state_dict(), directory + '/model.pth')
    pickle.dump(params, open(directory + '/params.pkl', 'wb'))
    pickle.dump(eval_optimization, open(directory + '/eval_optimization.pkl', 'wb'))

    # W_epoch = {'W_epoch': W}
    # pickle.dump(W_epoch, open(directory + '/W_epoch.pkl', 'wb'))


def predict_single_layer(id_: str, custom_test_set: dict=None):
    """ Tests model an returns and saves predicted values.

        If the prediction set is not the training set, pass a custom_test_set dictionary containing:
            'time_begin', 'duration', 'batch_size'

        Returns and saves:
            ../model/eval_prediction.pkl
    """
    # Define device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data and parameters
    print('Status: Load and process data for prediction.')
    params = pickle.load(open('../models/' + id_ + '/params.pkl', 'rb'))
    if custom_test_set is None:
        data_pre = pre_process(params=params, artificial=params['artificial_signal'][0])
        batch_size = params['batch_size']
    else:
        data_pre = pre_process(params=params, custom_test_set=custom_test_set)
        batch_size = custom_test_set['batch_size']
    data_set = iEEG_SingleLayerSet(data_pre)
    data_generator = torch.utils.data.DataLoader(data_set, batch_size=batch_size, shuffle=False)

    # Make model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = models.SingleLayer(params)
    model.load_state_dict(torch.load('../models/' + id_ + '/model.pth', map_location=device))
    model = model.to(device)

    # Evaluate model
    model.eval()
    pred_all = []
    true_all = []

    print('Status: Start prediction with cuda = ' + str(torch.cuda.is_available()) + '.')
    with torch.no_grad():
        for x, y in data_generator:
            x, y = x.float().to(device), y.float().to(device)
            predictions = model(x)
            pred_all.append(predictions.cpu().numpy())
            true_all.append(y.cpu().numpy())
        pred_all, true_all = np.concatenate(pred_all), np.concatenate(true_all)

    # Save predictions to file
    print('Status: Save predictions to file.')
    eval_prediction = {'prediction': pred_all,
                       'true': true_all}
    pickle.dump(eval_prediction, open('../models/' + id_ + '/eval_prediction.pkl', 'wb'))

    return eval_prediction


def generate_signal(fs=None, length=None, weights=None):
    if fs is None:
        fs = 512
    if weights is None:
        weights = np.array([[0.3, 0.7, 0], [0.9, 0.1, 0], [0.1, 0.1, 0.8]])
    if length is None:
        length = 5

    n_nodes = weights.shape[0]
    samples = np.arange(int(length * fs))
    signals = np.zeros((n_nodes, len(samples)))
    rand_freqs = np.random.uniform(0.5, 20, size=(n_nodes,))
    rand_amp = np.random.normal(.5, 1/5, size=(n_nodes,))
    rand_phase = np.random.uniform(0, 2 * np.pi, size=(n_nodes,))

    for i in range(n_nodes):
        signals[i, :] = np.sin(2 * np.pi * rand_freqs[i] * samples / fs + rand_phase[i]) * rand_amp[i]

    return np.matmul(weights, signals)


def coupled_oscillator(t_length, fs, small_weights=False):
    # Define Parameters
    k = np.array([[0.0, 0.1, 0.1, 1.0, 0.0, 0.0],
                  [0.1, 0.0, 1.0, 0.1, 0.0, 0.0],
                  [0.1, 1.0, 0.0, 0.1, 0.0, 0.0],
                  [1.0, 0.1, 0.1, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                  [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]])
    if small_weights:
        k = k / 10
    dt = 1/fs
    m = np.array([0.1, 0.05, 0.01, 0.05, 0.05, 0.01])
    k_0 = 0.3
    init = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

    # Compute matrix A
    diag = np.diag(-(2 * k_0 + np.sum(k, axis=1)))
    A = k + diag

    # Eigenvalue decomposition to get principal modes
    evals, evecs = np.linalg.eig(A)  # evals are principal modes
    r = np.lib.scimath.sqrt(evals/m)
    w = r.imag  # Frequencies of principal modes

    # Compute signal
    samples = np.arange(t_length * fs)
    t_vec = samples / fs
    signals = np.zeros((k.shape[0], len(samples)))
    signals[:, :2] = np.stack([init] * 2).T
    for t in range(2, len(samples)):
        signals[:, t] = np.matmul(A, signals[:, t-1]) * dt**2 / m - signals[:, t-2] + 2 * signals[:, t-1]

    plt.figure()
    for i in range(k.shape[0]):
        plt.plot(t_vec, signals[i, :], label=str(i))
    plt.legend()

    plt.figure()
    sns.heatmap(np.corrcoef(signals), vmin=-1, vmax=1, cmap='seismic', annot=True)

    return signals.T


def least_squares(patient_id, time_begin, duration, fs):
    # Load data
    data_mat = loadmat('../data/' + patient_id + '_' + str(time_begin[0]) + 'h.mat')
    info_mat = loadmat('../data/' + patient_id + '_' + 'info.mat')
    fs_orig = float(info_mat['fs'])
    if len(time_begin) == 3:
        sample_begin = int(np.round(time_begin[1] * 60 * fs + time_begin[2] * fs))
    else:
        sample_begin = int(np.round(time_begin[1] * 60 * fs))
    sample_end = int(np.round(sample_begin + duration * fs))
    data = data_mat['EEG'][:, sample_begin:sample_end].transpose()

    # Resample
    data = signal.resample(data, num=int(data.shape[0] / fs_orig * fs), axis=0)
    data = data[:,[3,4,50,51]]

    W = np.zeros((data.shape[1], data.shape[1]))
    for i in range(data.shape[1]):
        X_idx = [k for k in range(data.shape[1])]
        X_idx.remove(i)
        y = data[:, i]
        X = data[:, X_idx]
        w = np.linalg.inv(X.transpose().dot(X)).dot(X.transpose()).dot(y)
        W[i, X_idx] = w

    return W




