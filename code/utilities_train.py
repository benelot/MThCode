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


def train_and_test(params: dict):
    train(params)
    predict(params['id_'])
    distance(params['id_'])


def test_train_set(id_: str):
    predict(id_, predict_train_set=True)
    distance(id_, predict_train_set=True)


def pre_process(id_: str=None, params: dict=None, train_portion=0.8, windowing=False):
    """ Loads and prepares iEEG data.

        Returns:
            train_set, test_set
        Or (if windowing=True):
            X_train, X_test
    """
    # Load and cut data
    if params is None:
        params = pickle.load(open('../models/' + id_ + '/params.pkl', 'rb'))
    data_mat = loadmat(params['path2data'] + params['patient_id'] + '_' + str(params['time_begin'][0]) + 'h.mat')
    info_mat = loadmat(params['path2data'] + params['patient_id'] + '_' + 'info.mat')
    fs = float(info_mat['fs'])
    sample_begin = int(params['time_begin'][1] * 60 * fs)
    sample_end = int(sample_begin + params['duration'] * fs)
    if params['visible_size'] != 'all':
        data = data_mat['EEG'][:params['visible_size'], sample_begin:sample_end].transpose()
    else:
        data = data_mat['EEG'][:, sample_begin:sample_end].transpose()

    # Resample to fs of 512 Hz
    if fs != 512:
        data = signal.resample(data, num=int(data.shape[0] / fs * 512), axis=0)

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
    elif params['normalization'] is not None:
        print('Error: No valid normalization method.')

    # To tensor
    data = torch.from_numpy(data)

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


# def train_and_validate(params):
#     """ Trains model with parameters params.
#
#         Saves:
#             ../model/model.pth
#             ../model/params.pkl
#             ../model/eval_optim.pkl
#     """
#     # Define device
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#     # Load data
#     data_set = pre_process(params=params)
#     if params['visible_size'] == 'all':
#         params['visible_size'] = data_set.shape[1]
#     data_set = iEEG_DataSet(data_set, params['window_size'])
#
#     batch_size = 20
#     validation_split = .2
#     shuffle = True
#
#     n_windows = data_set.shape[0] - params['window_size']
#     indices = list(range(n_windows))
#     split = int(np.floor(validation_split * n_windows))
#     if shuffle:
#         np.random.seed(42)
#         np.random.shuffle(indices)
#     train_indices, valid_indices = indices[split:], indices[:split]
#     train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
#     valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(valid_indices)
#
#     train_generator = torch.utils.data.DataLoader(data_set, batch_size=batch_size, sampler=train_sampler)
#     test_generator = torch.utils.data.DataLoader(data_set, batch_size=batch_size, sampler=valid_sampler)
#
#     # Make model
#     model = models.GeneralRNN(params)
#     model = model.to(device)
#
#     # Define training parameters
#     criterion = None
#     if params['loss_function'] == 'mae':
#         criterion = nn.L1Loss(reduction='none')
#     elif params['loss_function'] == 'mse':
#         criterion = nn.MSELoss(reduction='none')
#     else:
#         print('Error: No valid loss function.')
#
#     lr = params['lr']
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#
#     # Training
#     loss = None
#     epoch_loss = np.zeros([params['epochs'], model.visible_size])
#     epoch_grad_norm = np.zeros(params['epochs'])
#
#     start_time = time.time()
#
#     for epoch in range(params['epochs']):
#         for X, y in train_generator:
#             X, y = X.to(device), y.float().to(device)
#             optimizer.zero_grad()
#             prediction = model(X)
#             loss = criterion(prediction, y)
#             loss.mean().backward()  # torch.autograd.backward(loss.mean())
#             optimizer.step()
#         for p in model.parameters():
#             epoch_grad_norm[epoch] = p.grad.data.norm(2).item()
#         epoch_loss[epoch, :] = np.mean(loss.detach().numpy(), axis=0)
#         # if epoch % 20 == 0:
#         print(f'Epoch: {epoch} | Loss: {np.mean(epoch_loss[epoch, :]):.4}')
#
#     total_time = time.time() - start_time
#     print(f'Time [min]: {total_time / 60:.3}')
#
#     # Make optimizer evaluation dictionary
#     eval_optimization = {'id_': params['id_'],
#                          'loss': epoch_loss,
#                          'grad_norm': epoch_grad_norm}
#
#     # Save trained model
#     directory = '../models/' + params['id_']
#     if not os.path.exists(directory):
#         os.mkdir(directory)
#     torch.save(model.state_dict(), directory + '/model.pth')
#     pickle.dump(params, open(directory + '/params.pkl', 'wb'))
#     pickle.dump(eval_optimization, open(directory + '/eval_optimization.pkl', 'wb'))
#
#     # Evaluate model
#     model.eval()
#     pred_all = []
#     true_all = []
#
#     with torch.no_grad():
#         for X, y in valid_generator:
#             X, y = X.to(device), y.to(device)
#             predictions = model(X)
#             pred_all.append(predictions)
#             true_all.append(y)
#         pred_all, true_all = np.concatenate(pred_all), np.concatenate(true_all)


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
    train_set, _ = pre_process(params=params)
    if params['visible_size'] == 'all':
        params['visible_size'] = train_set.shape[1]
    data_set = iEEG_DataSet(train_set, params['window_size'])
    data_generator = torch.utils.data.DataLoader(data_set, batch_size=200, shuffle=True)

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

    start_time = time.time()
    print('Status: Start training with cuda = ' + str(torch.cuda.is_available()) + '.')

    for epoch in range(params['epochs']):
        for X, y in data_generator:
            X, y = X.to(device), y.float().to(device)
            optimizer.zero_grad()
            prediction = model(X)
            loss = criterion(prediction, y)
            loss.mean().backward()  # torch.autograd.backward(loss.mean())
            optimizer.step()
        for p in model.parameters():
            epoch_grad_norm[epoch] = p.grad.data.norm(2).item()
        epoch_loss[epoch, :] = np.mean(loss.detach().numpy(), axis=0)
        if epoch % 20 == 0:
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


def predict(id_, predict_train_set=False):
    """ Tests model an returns and saves predicted values.

        Returns and saves:
            ../model/eval_prediction.pkl
    """
    # Define device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data and parameters
    params = pickle.load(open('../models/' + id_ + '/params.pkl', 'rb'))
    train_set, test_set = pre_process(params=params)
    if predict_train_set is True:
        data_set = iEEG_DataSet(train_set, params['window_size'])
        prefix = 'train'
    else:
        data_set = iEEG_DataSet(test_set, params['window_size'])
        prefix = 'test'
    data_generator = torch.utils.data.DataLoader(data_set, batch_size=200, shuffle=False)

    # Make model
    model = models.GeneralRNN(params)
    model.load_state_dict(torch.load('../models/' + id_ + '/model.pth'))
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
            pred_all.append(predictions)
            true_all.append(y)
        pred_all, true_all = np.concatenate(pred_all), np.concatenate(true_all)

    # Save predictions to file
    directory = '../models/' + id_ + '/eval_prediction.pkl'
    if os.path.exists(directory):
        eval_prediction = pickle.load(open(directory, 'rb'))
        eval_prediction[prefix + '_pred'] = pred_all
        eval_prediction[prefix + '_true'] = true_all
    else:
        eval_prediction = {'id_': id_,
                           prefix + '_pred': pred_all,
                           prefix + '_true': true_all}
    pickle.dump(eval_prediction, open('../models/' + id_ + '/eval_prediction.pkl', 'wb'))

    return eval_prediction


def distance(id_: str, predict_train_set=False):
    """ Computes Correlation, MSE, MAE for evaluation.

        Returns and saves:
            ../model/eval_distances.pkl
    """
    # Load parameters
    params = pickle.load(open('../models/' + id_ + '/params.pkl', 'rb'))
    eval_prediction = pickle.load(open('../models/' + id_ + '/eval_prediction.pkl', 'rb'))

    node_size = params['visible_size']

    if predict_train_set:
        pred = eval_prediction['train_pred']
        true = eval_prediction['train_true']
        train_str = '_train'
    else:
        pred = eval_prediction['test_pred']
        true = eval_prediction['test_true']
        train_str = ''

    # Calculate distances
    print('Status: Get distance metrics.')
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

    if predict_train_set is True:
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
