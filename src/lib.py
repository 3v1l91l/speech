from pathlib import Path
import pandas as pd
from scipy.io import wavfile
import numpy as np
from scipy import signal
import librosa
import matplotlib.pyplot as plt
from tqdm import tqdm

train_words = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go', 'silence']

def get_path_label_df(path, pattern='**/*.wav'):
    ''' Returns dataframe with columns: 'path', 'word'.'''
    datadir = Path(path)
    files = [(str(f), f.parts[-2]) for f in datadir.glob(pattern) if f]
    df = pd.DataFrame(files, columns=['path', 'word'])

    return df

def prepare_data(df):
    '''
    Remove _background_noise_ and replace not trained labels with unknown.
    '''
    words = df.word.unique().tolist()
    unknown = [w for w in words if w not in train_words]
    df = df.drop(df[df.word.isin(['_background_noise_'])].index)
    df.reset_index()
    df.loc[df.word.isin(unknown), 'word'] = 'unknown'
    return df

def get_specgrams(paths):
    len_paths = len(paths)
    log_specgrams = [None] * len_paths
    fs = 16000
    duration = 1
    for p in range(len_paths):
        wav, s = librosa.load(paths[p])
        if wav.size < fs:
            wav = np.pad(wav, (fs * duration - wav.size, 0), mode='constant')
        else:
            wav = wav[0:fs * duration]
        log_specgrams[p] = log_specgram(wav, fs)[..., np.newaxis]
        # log_specgrams[p] = spectrogram(wav, fs)[..., np.newaxis]

    # log_specgrams = [s.reshape(s.shape[0], s.shape[1], -1) for s in log_specgrams]
    return log_specgrams


def log_melspectrogram(wav, fs):
    windows_samples = int(fs / 40)
    hop_samples = int(windows_samples/5)
    melspec = librosa.feature.melspectrogram(wav, n_mels=128, sr=fs, fmax=8000, n_fft=windows_samples, hop_length=hop_samples, power=2.0)
    logspec = librosa.logamplitude(melspec)
    return logspec

def spectrogram(wav, fs):
    return signal.spectrogram(wav, fs=fs, nperseg=256, noverlap=128)[2]

def log_specgram(audio, sample_rate=16000, window_size=20,
                 step_size=10, eps=1e-10):
    nperseg = int(round(window_size * sample_rate / 1e3))
    noverlap = int(round(step_size * sample_rate / 1e3))
    freqs, times, spec = signal.spectrogram(audio,
                                    fs=sample_rate,
                                    window='hann',
                                    nperseg=nperseg,
                                    noverlap=noverlap,
                                    detrend=False)
    return np.log(spec.T.astype(np.float32) + eps)