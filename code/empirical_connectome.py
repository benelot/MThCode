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
"""
model_type = [['in', '57h', '58m', int(58*60*1024), int(1*60*1024)],
              ['in', '60h', '14m', int(14*60*1024), int(1*60*1024)],
              ['in', '64h', '40m', int(40*60*1024), int(1*60*1024)],
              ['is', '57h', '58m', int(58*60*1024), int(1*60*1024)],
              ['is', '60h', '14m', int(14*60*1024), int(1*60*1024)],
              ['is', '64h', '40m', int(40*60*1024), int(1*60*1024)]]
"""
ieeg.emp_connectome(patient_ID='07', hour='29to39_all', sperseg=120, soverlap=20)
