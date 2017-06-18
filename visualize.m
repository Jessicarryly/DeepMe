% ECG recordings were sampled as 300 Hz and they have been band pass filtered by the AliveCor device
clear all; close all; clc;
file = 'validation/A00011';
fs = 300;
N = 2048;
data = load(file); % load struct from .mat file
x = getfield(data, 'val'); % get field from struct
x = x(1:2048); % only take 2048 element
plot(x);title('Before filter')

analysis(x, fs)

% Filter design, allow 40 Hz to pass only
% y = sgolayfilt(x, 0, 5);
fc = 40; 
N = 41;
Wc = fc / (fs/2);
b = fir1(N, Wc, 'low', hamming(N+1));
y = filter(b, 1, x);
figure;
plot(y);title('After filter');
analysis(y, fs)