from scipy.io import loadmat
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from mne.time_frequency import tfr_multitaper
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator

# Import detections
channel_labels = ['ch00', 'ch10', 'ch20', 'ch30', 'ch40', 'ch50', 'ch60']
det_dict = np.load('../data/ID02_N03_detections.npy', allow_pickle=True).item()
spindles = {}
slow_waves = {}
for i, val in enumerate(channel_labels):
    spindles[val] = det_dict[val, 'Spindles']['index']/102.4
    slow_waves[val] = det_dict[val, 'Slow waves']['index']/102.4

# Import signal
data = loadmat('../data/ID02_55to63h.mat')
info = loadmat('../data/ID02_info.mat')
data = data['EEG']
fs = float(info['fs'])

# Signal processing
window = 120
f_bands = np.array([[0.5, 4], [4, 8], [8, 12], [12, 15]])

hypnogram_probs = []
for i, val in enumerate(channel_labels):
    print('Computing hypnogram: ' + str(i))
    # Make spectrogram
    f, _, Sxx = signal.spectrogram(data[i, :], fs, nperseg=int(window * fs), noverlap=0)

    # Make frequency band mask
    f_masked = []
    for k in range(f_bands.shape[0]):
        f_masked.append(np.zeros(Sxx.shape))
        f_bands_re = (f_bands * Sxx.shape[0] / f[-1]).astype(int)
        f_masked[k][f_bands_re[k, 0]:f_bands_re[k, 1], :] = 1
        f_masked[k] = np.multiply(f_masked[k], Sxx)

    f_window_integrated = np.zeros((f_bands.shape[0], Sxx.shape[1]))
    for k in range(f_bands.shape[0]):
        for j in range(Sxx.shape[1]):

            f_window_integrated[k, j] = np.sum(f_masked[k][:, j])

    # Make hypnogram probabilities
    lower_thres = np.mean([f_window_integrated[0, :].mean(), f_window_integrated[0, :].min()])
    upper_thres = np.mean([f_window_integrated[0, :].mean(), f_window_integrated[0, :].max()])
    thresholds = [lower_thres, upper_thres]

    delta_power = np.digitize(f_window_integrated[0, :], thresholds)
    spindles_mean = np.mean(spindles[val], axis=1)
    slow_waves_mean = np.mean(slow_waves[val], axis=1)

    hypnogram_prob = np.zeros((Sxx.shape[1]))
    for j in range(Sxx.shape[1]):
        # Delta power band
        hypnogram_prob[j] = delta_power[j] / len(thresholds)
        # Spindles occurred
        t_begin = j * window
        t_end = t_begin + window
        if len(spindles_mean[(spindles_mean > t_begin) & (spindles_mean < t_end)]) > 0:
            hypnogram_prob[j] = hypnogram_prob[j] + 0.25
        # Slow waves occurred
        if len(slow_waves_mean[(slow_waves_mean > t_begin) & (slow_waves_mean < t_end)]) > 0:
            hypnogram_prob[j] = hypnogram_prob[j] + 0.25
    hypnogram_probs.append(hypnogram_prob)


hypnogram_mean = np.mean(np.array(hypnogram_probs), axis=0)
hypnogram = np.where(hypnogram_mean > 0.4, 1, 0)

# Plot
fig, ax = plt.subplots(3, 1, figsize=(12, 8))
f, t, Sxx = signal.spectrogram(data[0, :], fs, nperseg=int(30*fs))
tvec = np.linspace(0, t[-1]/60/60, len(t))
ax[0].pcolormesh(tvec, f, Sxx, vmin=0, vmax=20)
ax[0].set_ylabel('Frequency [Hz]')
ax[0].set_ylim(0, 30)
t_hyp = np.arange(len(hypnogram))*window/3600
ax[1].step(t_hyp, hypnogram)
ax[1].set_xlim(0, t_hyp[-1])
ax[1].set_ylim(-.2, 1.75)
ax[1].scatter(spindles_mean/60/60, np.ones(len(spindles_mean))*1.25, color='tab:red', marker='v')
ax[1].scatter(slow_waves_mean/60/60, np.ones(len(slow_waves_mean))*1.5, color='tab:orange', marker='d')
ax[1].set_yticks([0, 1, 1.25, 1.5])
ax[1].set_yticklabels(['Wake', 'NREM', 'Spindles', 'Slow waves'])
ax[2].plot(t_hyp, hypnogram_mean)
ax[2].set_xlim(0, t_hyp[-1])
ax[2].set_ylim(0, 1)
ax[2].set_ylabel('Probability')
ax[2].set_xlabel('Time [h]')
plt.show()
#plt.savefig('Sleep_scoring.png')
