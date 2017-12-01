from pathlib import Path
import pandas as pd
from scipy.io import wavfile
import numpy as np
from scipy import signal
import librosa
import matplotlib.pyplot as plt
from tqdm import tqdm
from glob import glob
import os
import math
import random

L = 16000
root_path = r'..'
out_path = r'.'
model_path = r'.'
train_data_path = os.path.join(root_path, 'input', 'train', 'audio')
test_data_path = os.path.join(root_path, 'input', 'test', 'audio')
background_noise_paths = glob(os.path.join(train_data_path, r'_background_noise_/*' + '.wav'))
silence_paths = glob(os.path.join(train_data_path, r'silence/*' + '.wav'))

legal_labels = 'yes no up down left right on off stop go silence unknown'.split()

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
    unknown = [w for w in words if w not in legal_labels]
    df = df.drop(df[df.word.isin(['_background_noise_'])].index)
    df.reset_index(inplace=True)
    df.loc[df.word.isin(unknown), 'word'] = 'unknown'
    return df

def get_specgrams(wavs):
    log_specgrams = [None] * len(wavs)
    fs = 16000
    for i in range(len(wavs)):
        log_specgrams[i] = log_specgram(wavs[i], fs)[..., np.newaxis]
    return log_specgrams

def get_specgrams_augment_known(wavs, silences, unknowns):
    len_paths = len(wavs)
    log_specgrams = [None]*len_paths
    fs = 16000
    for i in range(len(wavs)):
        wav = wavs[i]
        noise = silences[random.randint(0, len(silences)-1)]
        scale = np.random.uniform(low=0, high=0.3, size=1)
        wav = (1 - scale) * wav + (noise * scale)
        log_specgrams[i] = log_specgram(wav, fs)[..., np.newaxis]
    return log_specgrams

def get_specgrams_augment_unknown(wavs, silences, unknowns):
    len_paths = len(wavs)
    log_specgrams = [None]*len_paths
    fs = 16000
    duration = 1
    for i in range(len(wavs)):
        wav = wavs[i]
        for x in np.random.randint(0, fs, 4):
            unknown_overlap = unknowns[random.randint(0, len(unknowns)-1)]
            unknown_overlap = pad(unknown_overlap, fs, duration)
            wav = (1 - 0.5) * wav + (np.concatenate((unknown_overlap[x:],unknown_overlap[:x])) * 0.5)
        noise = silences[random.randint(0, len(silences)-1)]
        scale = np.random.uniform(low=0, high=0.3, size=1)
        wav = (1 - scale) * wav + (noise * scale)
        log_specgrams[i] = log_specgram(wav, fs)[..., np.newaxis]
    return log_specgrams

def pad(wav, fs, duration):
    if wav.size < fs:
        wav = np.pad(wav, (fs * duration - wav.size, 0), mode='constant')
    else:
        wav = wav[0:fs * duration]
    return wav

def log_melspectrogram(wav, fs):
    windows_samples = int(fs / 40)
    hop_samples = int(windows_samples/5)
    melspec = librosa.feature.melspectrogram(wav, n_mels=128, sr=fs, fmax=8000, n_fft=windows_samples, hop_length=hop_samples, power=2.0)
    logspec = librosa.logamplitude(melspec)
    return logspec

def spectrogram(wav, fs):
    return signal.spectrogram(wav, fs=fs, nperseg=256, noverlap=128)[2]

# def log_specgram(audio, sample_rate=16000, window_size=20,
#                  step_size=10, eps=1e-10):
#     nperseg = int(round(window_size * sample_rate / 1e3))
#     noverlap = int(round(step_size * sample_rate / 1e3))
#     freqs, times, spec = signal.spectrogram(audio,
#                                     fs=sample_rate,
#                                     window='hann',
#                                     nperseg=nperseg,
#                                     noverlap=noverlap,
#                                     detrend=False)
#     return np.log(spec.T.astype(np.float32) + eps)

def log_specgram(data, sr=16000):
    data = librosa.feature.melspectrogram(data, sr=sr, n_mels=40, hop_length=160, n_fft=480, fmin=20, fmax=4000)
    data[data > 0] = np.log(data[data > 0])
    data = [np.matmul(librosa.filters.dct(40,40), x) for x in np.split(data, data.shape[1], axis=1)]
    data = np.array(data, order="F").squeeze(2).astype(np.float32)
    return data

def label_transform(labels):
    nlabels = []
    for label in labels:
        if label not in legal_labels:
            nlabels.append('unknown')
        else:
            nlabels.append(label)
    return pd.get_dummies(pd.Series(nlabels))

def load_wav_by_path(p):
    _, wav = wavfile.read(p)
    if wav.size < L:
        wav = np.pad(wav, (L - wav.size, 0), mode='constant')
    else:
        wav = wav[0:L]
    return wav

