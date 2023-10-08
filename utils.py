import os
import copy
import random
import numpy as np
import matplotlib
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from scipy.fftpack import fft, ifft
from scipy.fft import rfft
import torch

matplotlib.use('Agg')
sns.set_theme()
sns.set_style("white")
sns.set_palette("bright")


def to_device(models, device):
    for model in models:
        model = model.to(device)


def filter(X, sampling_rate=256):
    fkernB, fkernA = signal.butter(7, 50, btype='lowpass', fs=sampling_rate)
    X = signal.filtfilt(fkernB, fkernA, X)
    return X


def disable_splines(ax, splines=['top', 'right', 'bottom', 'left']):
    for spline in splines:
        ax.spines[spline].set_visible(False)


def hide_first_stick(ax):
    def func(x, pos): return 1 if np.isclose(x, 0) else int(x)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(func))


def plot_23channels_oop(eeg, sampling_rate=256, filt=True, T=10):
    for i in range(T-2):
        a_signal = interpolate_eeg_boundaries(
            eeg[:, sampling_rate * i: sampling_rate * (i + 2)])
        eeg[:, sampling_rate * i: sampling_rate * (i + 2)] = a_signal

    fig, axs = plt.subplots(5, 2, figsize=(
        6, 6), constrained_layout=True, gridspec_kw={'width_ratios': [3, 1]})
    channels = [12,  0,  8,  4,  1]
    channel_names = ['FP2-F8', 'FP1-F7', 'FP2-F4', 'FP1-F3', 'F7-T7']

    for idx, ax in enumerate(axs[:, 0]):
        ax.set_ylabel(channel_names[idx], fontsize=10)

    for idx, ch in enumerate(channels):
        sig = eeg[idx, :]
        if filt:
            sig = filter(sig)
        time = np.arange(0, len(sig)) / sampling_rate
        axs[idx][0].plot(time, sig, color='black', linewidth=0.5)

        disable_splines(axs[idx][0], ['top', 'bottom', 'left'])
        axs[idx][0].get_yaxis().set_ticks([])
        if idx < 4:
            axs[idx][0].get_xaxis().set_ticklabels([])

        fourier_transform = rfft(sig)
        abs_fourier_transform = np.abs(fourier_transform)

        frequency = np.linspace(0, sampling_rate/2, len(abs_fourier_transform))
        half_len = int(len(frequency)/2)
        amp_spectrum = abs_fourier_transform[1:half_len]
        amp_spectrum = signal.savgol_filter(amp_spectrum, 11, 2)

        axs[idx][1].fill_between(
            frequency[1:half_len], amp_spectrum, color='purple', alpha=0.4)
        disable_splines(axs[idx][1])
        axs[idx][1].get_yaxis().set_ticks([])
        if idx < 4:
            axs[idx][1].get_xaxis().set_ticklabels([])

        if idx == 4:
            axs[idx][0].set_xlabel('Time (s)', fontsize=10)
            axs[idx][0].xaxis.set_tick_params(labelsize=10)
            axs[idx][1].set_xlabel('Frequency (hz)', fontsize=10)
            axs[idx][1].xaxis.set_tick_params(labelsize=10)

    plt.subplots_adjust(wspace=0, hspace=-0.1)


def plot_freq_domain(valid_data, gen_samples, sampling_rate, img_dir):
    plt.figure(figsize=(10, 5))

    fourier_transform = np.fft.rfft(valid_data)
    abs_fourier_transform = np.abs(fourier_transform)
    amp_spectrum = abs_fourier_transform
    amp_spectrum_val = np.mean(amp_spectrum, axis=0)

    fourier_transform = np.fft.rfft(gen_samples)
    abs_fourier_transform = np.abs(fourier_transform)
    amp_spectrum = abs_fourier_transform
    amp_spectrum_gen = np.mean(amp_spectrum, axis=0)

    frequency = np.linspace(0, sampling_rate/2, len(amp_spectrum_gen))
    plt.plot(frequency[1:], 20*np.log10(amp_spectrum_val[1:]), label='Ref.')
    plt.plot(frequency[1:], 20*np.log10(amp_spectrum_gen[1:]), label='Syn.')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Log Magnitude')
    plt.title('Mean frequency spectra')
    plt.legend()

    plt.savefig(img_dir, dpi=200)
    plt.close()


def plot_time_domain(valid_data, gen_samples, img_dir):
    mean_valid = np.mean(valid_data.numpy(), axis=0)
    std_valid = np.std(valid_data.numpy(), axis=0)
    plt.plot(mean_valid, label='Ref.')
    plt.fill_between(range(len(mean_valid)), mean_valid -
                     std_valid, mean_valid+std_valid, alpha=.3)

    mean_gen = np.mean(gen_samples.numpy(), axis=0)
    std_gen = np.std(gen_samples.numpy(), axis=0)
    plt.plot(mean_gen, label='Syn.')
    plt.fill_between(range(len(mean_gen)), mean_gen -
                     std_gen, mean_gen+std_gen, alpha=.3)

    plt.xlabel('Time (10s - 256Hz)')
    plt.title(
        'Distribution of values at each time point')
    plt.legend()

    plt.savefig(img_dir, dpi=200)
    plt.close()


def interpolate_eeg_boundaries(eeg, sampling_rate=256):
    signal1 = copy.deepcopy(eeg)

    boundaryPnts = [
        sampling_rate - int(sampling_rate / 10), sampling_rate + int(sampling_rate / 10)]
    signal1[:, range(boundaryPnts[0], boundaryPnts[1])] = np.nan

    fftPre = fft(
        signal1[:, range(boundaryPnts[0]-int(np.diff(boundaryPnts)), boundaryPnts[0])])
    fftPst = fft(signal1[:, range(boundaryPnts[1]+1,
                 boundaryPnts[1]+int(np.diff(boundaryPnts)+1))])

    mixeddata = signal.detrend(np.real(ifft((fftPre+fftPst)/2)))
    linedata = np.repeat(np.linspace(0, 1, int(np.diff(boundaryPnts)))[np.newaxis, :], repeats=len(signal1), axis=0) * \
        (signal1[:, boundaryPnts[1]+1]-signal1[:, boundaryPnts[0]-1])[:, np.newaxis] + \
        signal1[:, boundaryPnts[0]-1][:, np.newaxis]

    linterp = mixeddata + linedata

    filtsig = copy.deepcopy(signal1)
    filtsig[:, range(boundaryPnts[0], boundaryPnts[1])] = linterp

    return filtsig


def set_seed(seed=3013):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")
