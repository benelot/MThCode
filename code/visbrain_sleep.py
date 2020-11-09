from scipy.io import loadmat
from visbrain.gui import Sleep
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# Load the matlab file :
mat = loadmat('../data/ID11_59to65h.mat')
info = loadmat('../data/ID11_info.mat')
# Prepare the data :
sf = float(info['fs'])
data = mat['EEG']
channel_labels = ['ch00', 'ch20', 'ch40', 'ch60']

# Open the GUI :
Sleep(data=data, sf=sf, channels=channel_labels).show()
