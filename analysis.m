% ECG recordings were sampled as 300 Hz and they have been band pass filtered by the AliveCor device
clear all; close all; clc;
file = 'validation/A00585';
fs = 300;
N = 2048;
data = load(file); % load struct from .mat file
x = getfield(data, 'val'); % get field from struct
x = x(1:2048); % only take 2048 element
plot(x);

% One-side Spectral analysis
X = abs(fft(x)) / N;
X = X(1:N/2 + 1)
ps = X.^2;
f = [0:N/2] * fs/N;
figure;
subplot(211);plot(f, X);xlabel('Frequency (Hz)');ylabel('Amplitude spectrum');
subplot(212);plot(f, ps);xlabel('Frequency (Hz)');ylabel('Power spectrum');
