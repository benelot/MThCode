clc, close all , clear

data_id11 = load('../data/ID11_all_fs64.mat');
x11 = data_id11.data;
data_id12 = load('../data/ID12_all_fs64.mat');
x12 = data_id12.data;
fs = 64;

set(0,'defaulttextInterpreter','latex') 
set(0,'DefaultLegendInterpreter','latex')
set(groot,'defaultAxesTickLabelInterpreter','latex');  

subplot(211)
spectrogram(x11, 2*60*fs, 0, 2*60*fs, fs, 'yaxis')
h = colorbar;
h.Label.String = 'Power density (dB/Hz)';
h.Label.Interpreter = 'latex';
caxis([0, 40])
ylim([0, 10])

subplot(212)
spectrogram(x12, 2*60*fs, 0, 2*60*fs, fs, 'yaxis')
h = colorbar;
h.Label.String = 'Power density (dB/Hz)';
h.Label.Interpreter = 'latex';
caxis([0, 40])
ylim([0, 10])