from pathlib import Path
import pandas as pd
from scipy.io import wavfile
import numpy as np
from scipy import signal
import librosa
import matplotlib.pyplot as plt

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

def windows(data, window_size):
    start = 0
    while start < len(data):
        yield start, start + window_size
        start += (window_size / 2)

def get_specgrams(paths, bands = 60, frames = 61):
    log_specgrams = []
    fs = 16000
    duration = 1
    for path in paths:
        wav,s = librosa.load(path)
        if wav.size < fs:
            wav = np.pad(wav, (fs * duration - wav.size, 0), mode='constant')
        else:
            wav = wav[0:fs * duration]
        log_specgrams.append(librosa.logamplitude(np.abs(librosa.core.stft(wav, win_length=400, hop_length=160,center=False)), ref_power=np.max))
        log_specgrams = [s.reshape(s.shape[0], s.shape[1], -1) for s in log_specgrams]
    return log_specgrams

def get_specgrams(paths, shape, duration=1):
    '''
    Given list of paths, return specgrams.
    '''
    fs = 16000
    wavs = [wavfile.read(x)[1] for x in paths]
    data = []
    for wav in wavs:
        if wav.size < 16000:
            d = np.pad(wav, (fs*duration - wav.size, 0), mode='constant')
        else:
            d = wav[0:fs*duration]
        data.append(d)

    # specgram = [log_specgram(d, fs) for d in data]
    # specgram = [s.reshape(shape[0], shape[1], -1) for s in specgram]
    specgram = [signal.spectrogram(d, nperseg=256, noverlap=128)[2] for d in data]
    specgram = [s.reshape(129, 124, -1) for s in specgram]
    return specgram