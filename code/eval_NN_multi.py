""" Comparison of various NNs

Part of master thesis Segessenmann J. (2020)
"""
import utilities as util
import models
import pandas as pd
import torch
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

id = ['test_0', 'test_1', 'test_2']

# Todo: Make this in function (+ single function seperately?)
eval_distances = pickle.load(open('../models/' + id[0] + '/eval_distances.pkl', 'rb'))
df = pd.DataFrame(eval_distances)
for i in range(len(id)-1):
    eval_distances = pickle.load(open('../models/' + id[i+1] + '/eval_distances.pkl', 'rb'))
    df = df.append(pd.DataFrame(eval_distances), ignore_index=True)

print(df.head())
util.plot_multi_boxplots(df, x='ID', y='Correlation', ylim=(0, 1))

# Todo: Make this in function (+ single function seperately?)
eval_prediction0 = pickle.load(open('../models/' + id[0] + '/eval_prediction.pkl', 'rb'))
eval_prediction1 = pickle.load(open('../models/' + id[1] + '/eval_prediction.pkl', 'rb'))
eval_prediction2 = pickle.load(open('../models/' + id[2] + '/eval_prediction.pkl', 'rb'))
# util.plot_multi_scatter(id, )