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

def log_specgram(audio, sample_rate, window_size=20,
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

    # # audio, sample_rate = librosa.load(audio_path, sr=None)
    # s = librosa.logamplitude(librosa.feature.melspectrogram(audio, sr=sample_rate, n_mels=96), ref_power=1.0)
    # plt.imshow(s)
    # plt.show()
    # return s

def get_specgrams(paths, shape, duration=1):
    '''
    Given list of paths, return specgrams.
    '''
    fs = 16000
    wavs = [wavfile.read(x)[1] for x in paths]
    data = []
    for wav in wavs:
        if wav.size < fs:
            d = np.pad(wav, (fs * duration - wav.size, 0), mode='constant')
        else:
            d = wav[0:fs * duration]
        data.append(d)

    specgram = [log_specgram(d, fs) for d in data]
    specgram = [s.reshape(shape[0], shape[1], -1) for s in specgram]
    # specgram = [signal.spectrogram(d, nperseg=256, noverlap=128)[2] for d in data]
    # specgram = [s.reshape(129, 124, -1) for s in specgram]
    return specgram