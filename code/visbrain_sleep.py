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
"""
f, t, Sxx = signal.spectrogram(data, sf, nperseg=int(30*sf))
tvec = np.linspace(0, t[-1]/60/60, len(t))
plt.locator_params(axis='x', nbins=40)
plt.pcolormesh(tvec, f, Sxx, vmin=0, vmax=20)
plt.xlabel('Time [h]')
plt.ylabel('Frequency [Hz]')
plt.ylim(0, 30)
plt.colorbar()
plt.show()
"""
# Open the GUI :
Sleep(data=data, sf=sf, channels=channel_labels).show()
