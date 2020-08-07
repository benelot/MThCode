import numpy as np
import pickle


def print_params(id_: str):
    params = pickle.load(open('../models/' + id_ + '/params.pkl', 'rb'))
    for keys, values in params.items():
        print(f'{keys}: {values}')


def change_params(id_: str, param: str, val):
    params = pickle.load(open('../models/' + id_ + '/params.pkl', 'rb'))
    params[param] = val
    pickle.dump(params, open('../models/' + id_ + '/params.pkl', 'wb'))


def change_params_key(id_: str, old_key: str, new_key: str):
    params = pickle.load(open('../models/' + id_ + '/params.pkl', 'rb'))
    val = params[old_key]
    del params[old_key]
    params[new_key] = val
    pickle.dump(params, open('../models/' + id_ + '/params.pkl', 'wb'))


def new_params(id_: str, key: str, val):
    params = pickle.load(open('../models/' + id_ + '/params.pkl', 'rb'))
    params[key] = val
    pickle.dump(params, open('../models/' + id_ + '/params.pkl', 'wb'))

