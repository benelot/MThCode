import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

import utilities as util


# Load parameters
params = pickle.load(open('../models/FRNN_tanh_big.pkl', 'rb'))  # rotearlyopt normal

# Load data
train_set, test_set = util.data_loader(params, windowing=False)

df = pd.DataFrame(train_set)
corr = df.corr()

fig, ax = plt.subplots()
plt.title('Channel correlation matrix')
sns.heatmap(corr, cmap='RdBu', vmin=-1, vmax=1, square=True)
plt.show()
fig.savefig('../doc/figures/corr_tanh_big.png')
