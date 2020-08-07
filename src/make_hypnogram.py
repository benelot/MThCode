from scipy.io import loadmat
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
#from mne.time_frequency import tfr_multitaper
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator

# Import detections
welchs_method = False
delta_wave_quantile_thres = .6
event_min_max_thres = 0.2
show_channel = 'ch00'

channel_labels = ['ch00', 'ch10', 'ch20', 'ch30', 'ch40', 'ch50', 'ch60']
det_dict = np.load('../data/ID02_N03_detections(1).npy', allow_pickle=True).item()
spindles = {}
slow_waves = {}
k_complexes = {}
for i, val in enumerate(channel_labels):
    spindles[val] = det_dict[val, 'Spindles']['index']/102.4
    slow_waves[val] = det_dict[val, 'Slow waves']['index']/102.4
    k_complexes[val] = det_dict[val, 'K-complexes']['index']/102.4

# Import signal
data = loadmat('../data/ID02_55to63h.mat')
info = loadmat('../data/ID02_info.mat')
data = data['EEG']
fs = float(info['fs'])

# Signal processing
window = int(2*60*fs)
f_bands_hz = np.array([[0.5, 4], [4, 8], [8, 12], [12, 15]])

hypnograms = []
for i, val in enumerate(channel_labels):
    print('Computing hypnogram: ' + str(i))
    # Make spectrogram
    fvec_S, _, Sxx = signal.spectrogram(data[i, :], fs, nperseg=window, noverlap=0)
    f_bands_re, spectr = (f_bands_hz * Sxx.shape[0] / fvec_S[-1]).astype(int),  Sxx

    if welchs_method is True:
        nperseg_welch = 256
        Pxx = np.zeros((int(nperseg_welch/2 + 1), Sxx.shape[1]))
        for t in range(Sxx.shape[1]):
            fvec_P, Pxx[:, t] = signal.welch(data[i, int(t*window):int(t*window + window)],
                                                 fs=fs, nperseg=nperseg_welch)
        f_bands_re, spectr = (f_bands_hz * Pxx.shape[0] / fvec_P[-1]).astype(int), Pxx

    # Compute power density per band
    band_power = np.zeros((f_bands_re.shape[0], Sxx.shape[1]))
    for t in range(Sxx.shape[1]):
        for band in range(f_bands_re.shape[0]):
            band_power[band, t] = np.sum(spectr[f_bands_re[band, 0]:f_bands_re[band, 1], t])

    # Compute event density
    if spindles[val].size:
        spindles_tmean = np.mean(spindles[val], axis=1)
        spindles_idx = list(np.round(spindles_tmean*fs/window).astype(int))
        spindles_idx = list(filter(lambda a: a != Sxx.shape[1], spindles_idx))
        spindles_dens = np.zeros(Sxx.shape[1])
        for _, idx in enumerate(spindles_idx):
            spindles_dens[idx] = spindles_dens[idx] + 1
    if slow_waves[val].size:
        slow_waves_tmean = np.mean(slow_waves[val], axis=1)
        slow_waves_idx = list(np.round(slow_waves_tmean * fs / window).astype(int))
        slow_waves_idx = list(filter(lambda a: a != Sxx.shape[1], slow_waves_idx))  # removes idx + 1
        slow_waves_dens = np.zeros(Sxx.shape[1])
        for _, idx in enumerate(slow_waves_idx):
            slow_waves_dens[idx] = slow_waves_dens[idx] + 1
    if k_complexes[val].size:
        k_complexes_tmean = np.mean(k_complexes[val], axis=1)
        k_complexes_idx = list(np.round(k_complexes_tmean * fs / window).astype(int))
        k_complexes_idx = list(filter(lambda a: a != Sxx.shape[1], k_complexes_idx))  #
        k_complexes_dens = np.zeros(Sxx.shape[1])
        for _, idx in enumerate(k_complexes_idx):
            k_complexes_dens[idx] = k_complexes_dens[idx] + 1

    # Categorize sleep - wake/REM
    hypnogram = np.where(band_power[0, :] > np.quantile(band_power[0, :], delta_wave_quantile_thres), 1, 0)

    # Categorize N1 - N2 - N3
    if spindles[val].size and slow_waves[val].size:
        if k_complexes[val].size:
            n2n3_diff = spindles_dens / spindles_dens.max()- slow_waves_dens / slow_waves_dens.max()# + 0.3 * k_complexes_dens / k_complexes_dens.max()\

        else:
            n2n3_diff = spindles_dens / spindles_dens.max() - slow_waves_dens / slow_waves_dens.max()
        n2n3 = np.where(n2n3_diff > event_min_max_thres, 1, n2n3_diff)
        n2n3 = np.where(n2n3 < -event_min_max_thres, 2, n2n3)
        n2n3 = np.where(n2n3 < event_min_max_thres + 0.1, 0, n2n3)
        hypnogram = hypnogram + n2n3

    hypnograms.append(hypnogram)

    # hypnogram_prob = np.zeros((2, Sxx.shape[1]))
    # for t in range(Sxx.shape[1]):
    #     # Categorize sleep
    #     hypnogram_prob[0, t] = band_power[0, t]
    #     # Spindles occurred
    #     t_begin = j * window
    #     t_end = t_begin + window
    #     if len(spindles_mean[(spindles_mean > t_begin) & (spindles_mean < t_end)]) > 0:
    #         hypnogram_prob[j] = hypnogram_prob[j] + 0.25
    #     # Slow waves occurred
    #     if len(slow_waves_mean[(slow_waves_mean > t_begin) & (slow_waves_mean < t_end)]) > 0:
    #         hypnogram_prob[j] = hypnogram_prob[j] + 0.25

    # Save data from channel to show
    if val == show_channel:
        band_power_plt = band_power
        if spindles[val].size:
            spindles_plt = spindles_tmean.copy()
        if slow_waves[val].size:
            slow_waves_plt = slow_waves_tmean.copy()
        if k_complexes[val].size:
            k_complexes_plt = k_complexes_tmean.copy()

hypnograms = np.asarray(hypnograms).astype(int)
hypnogram = np.zeros(hypnograms.shape[1])
for t in range(hypnograms.shape[1]):
    hypnogram[t] = np.argmax(np.bincount(hypnograms[:, t]))
hypnogram = hypnogram*(-1)

# Plot
fig, ax = plt.subplots(3, 1, figsize=(12, 8))
fvec, t, Sxx = signal.spectrogram(data[0, :], fs, nperseg=window, noverlap=0)
tvec = np.linspace(0, t[-1]/60/60, len(t))
ax[0].pcolormesh(tvec, fvec, Sxx, vmin=0, vmax=20)
ax[0].set_ylabel('Frequency [Hz]')
ax[0].set_ylim(0, 30)
ax[0].set_title('Spectrogram of ' + show_channel)

t_hyp = np.arange(len(hypnogram))*window/3600/fs
ax[1].plot(t_hyp, band_power_plt[0, :]/10000)
ax[1].set_xlim(0, t_hyp[-1])
ax[1].set_ylabel('Delta power dens.')

ax[2].step(t_hyp, hypnogram)
ax[2].set_xlim(0, t_hyp[-1])
ax[2].set_ylim(-3.5, 2)
if spindles[val].size:
    ax[2].scatter(spindles_plt/60/60, np.ones(len(spindles_plt))*0.5, color='tab:red', marker='v')
if slow_waves[val].size:
    ax[2].scatter(slow_waves_plt/60/60, np.ones(len(slow_waves_plt))*1, color='tab:orange', marker='d')
if k_complexes[val].size:
    ax[2].scatter(k_complexes_plt/60/60, np.ones(len(k_complexes_plt))*1.5, color='tab:cyan', marker='X')
ax[2].set_yticks([-3, -2, -1, 0, .5, 1, 1.5])
ax[2].set_yticklabels(['N3', 'N2', 'N1', 'Wake/REM', 'Spindles', 'Slow waves', 'K-complexes'])
ax[2].set_xlabel('Time [h]')

plt.show()

