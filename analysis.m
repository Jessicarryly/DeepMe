function analysis(x, fs)
    % One-side Spectral analysis
    N = length(x);
    X = abs(fft(x)) / N;
    X = X(1:N/2 + 1);
    ps = X.^2;
    f = [0:N/2] * fs/N;
    figure;
    subplot(211);plot(f, X);xlabel('Frequency (Hz)');ylabel('Amplitude spectrum');
    subplot(212);plot(f, ps);xlabel('Frequency (Hz)');ylabel('Power spectrum');
end




