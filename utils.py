import librosa
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio.transforms as T
from scipy.signal import ellip, filtfilt, firwin, lfilter
from torchaudio.transforms import MelSpectrogram


# --- Preprocess Event ---
def get_event_cond(x, event_type='rms'):
    assert event_type in ['rms', 'power', 'onset']
    if event_type == 'rms':
        return get_rms(x)
    if event_type == 'power':
        return get_power(x)
    if event_type == 'onset':
        return get_onset(x)
    
def get_rms(signal):
    rms = librosa.feature.rms(y=signal, frame_length=512, hop_length=128)
    rms = rms[0]
    rms = zero_phased_filter(rms)
    return torch.tensor(rms.copy(), dtype=torch.float32)

def get_power(signal):
    if torch.is_tensor(signal):
        signal_copy_grad = signal.clone().detach().requires_grad_(signal.requires_grad)
        return signal_copy_grad*signal_copy_grad
    else:
        return torch.tensor(signal*signal, dtype=torch.float32)
    
def get_onset(y, sr=22050):
    y = np.array(y)
    o_env = librosa.onset.onset_strength(y=y, sr=sr, aggregate=np.median, fmax=8000, n_mels=256)
    onset_frames = librosa.onset.onset_detect(onset_envelope=o_env, sr=sr, normalize=True, delta=0.3, units='samples')
    onsets = np.zeros(y.shape)
    onsets[onset_frames] = 1.0
    return torch.tensor(onsets, dtype=torch.float32)

def resample_audio(audio, original_sr, target_sr):
    resampler = T.transforms.Resample(original_sr, target_sr, resampling_method='sinc_interpolation')
    return resampler(audio)

def adjust_audio_length(audio, length):
    if audio.shape[1] >= length:
        return audio[0, :length]
    return torch.cat((audio[0, :], torch.zeros(length - audio.shape[1])), dim=-1)


# --- Post-process Audio ---
def normalize(x):
    return x / torch.max(torch.abs(x)).item()

def high_pass_filter(x, sr=22050):
    b = firwin(101, cutoff=20, fs=sr, pass_zero='highpass')
    x= lfilter(b, [1,0], x)
    return x
    
def zero_phased_filter(x):
    b, a = ellip(4, 0.01, 120, 0.125) 
    x = filtfilt(b, a, x, method="gust")
    return x

def pooling(x):
    block_num = 490
    block_size = x.shape[-1] // block_num
    
    device = x.device
    pooling = torch.nn.MaxPool1d(block_size, stride=block_size)
    x = x.unsqueeze(1)
    pooled_x = pooling(x).to(device)
    
    return pooled_x


# --- Plot ---
def save_figure_to_numpy(fig):
    # save it to a numpy array.
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data
    
def plot_spec(waveform, sample_rate):
    # Transform to mel-spec
    transform = MelSpectrogram(sample_rate)
    mel_spec = transform(waveform)
    
    # Plot
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(mel_spec)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    
    # Turn into numpy format to upload to tensorboard
    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close()
    return data

def plot_env(waveform):
    # Plot
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(12, 3))
    plt.plot(waveform)
    plt.tight_layout()
    
    # Turn into numpy format to upload to tensorboard
    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close()
    return data