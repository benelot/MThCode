from scipy.io import loadmat
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib import animation, rc
from IPython.display import HTML, Image
import ieeg_processing as ieeg

# Import signal
channel_labels = ['55', '58', '59', '61', '64', '70']
for i, val in enumerate(channel_labels):
    ieeg.emp_connectome_anim('02', val, sperseg=60, soverlap=20)
