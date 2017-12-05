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
new_sample_rate = 8000
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
        # log_specgrams[i] = log_specgram(wavs[i], fs)[..., np.newaxis]
        log_specgrams[i] = log_specgram(wavs[i], fs)
    return log_specgrams

def get_specgrams_augment_known(wavs, silences, unknowns):
    len_paths = len(wavs)
    log_specgrams = [None]*len_paths
    fs = 16000
    for i in range(len(wavs)):
        # wav = wavs[i]
        # noise = silences[random.randint(0, len(silences)-1)]
        # scale = np.random.uniform(low=0, high=0.3, size=1)
        # wav = (1 - scale) * wav + (noise * scale)
        wav = augment_data(wavs[i], fs, silences)
        # log_specgrams[i] = log_specgram(wav, fs)[..., np.newaxis]
        log_specgrams[i] = log_specgram(wav, fs)
    return log_specgrams

def get_specgrams_augment_unknown(wavs, silences, unknowns):
    len_paths = len(wavs)
    log_specgrams = [None]*len_paths
    fs = 16000
    duration = 1
    for i in range(len(wavs)):
        # wav = wavs[i]
        # for x in np.random.randint(0, fs, 4):
        #     unknown_overlap = unknowns[random.randint(0, len(unknowns)-1)]
        #     unknown_overlap = pad(unknown_overlap, fs, duration)
        #     wav = (1 - 0.5) * wav + (np.concatenate((unknown_overlap[x:],unknown_overlap[:x])) * 0.5)
        # noise = silences[random.randint(0, len(silences)-1)]
        # scale = np.random.uniform(low=0, high=0.3, size=1)
        # wav = (1 - scale) * wav + (noise * scale)
        wav = augment_data(wavs[i], fs, silences)
        # log_specgrams[i] = log_specgram(wav, fs)[..., np.newaxis]
        log_specgrams[i] = log_specgram(wav, fs)
    return log_specgrams

def get_specgrams_aug(wav):
    fs = 16000
    return log_specgram(augment_data(wav, fs), fs)[..., np.newaxis]

def pad(wav, fs, duration):
    if wav.size < fs:
        wav = np.pad(wav, (fs * duration - wav.size, 0), mode='constant')
    else:
        wav = wav[0:fs * duration]
    return wav

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
#     data = np.log(spec.T.astype(np.float32) + eps)
#     mean = np.mean(np.ravel(data))
#     std = np.std(np.ravel(data))
#     if std != 0:
#         data = data - mean
#         data = data / std
#     return data


# def log_specgram(audio, sample_rate=16000, window_size=20,
#                  step_size=10, eps=1e-10):
#     # wave, sr = librosa.load(file_path, mono=True, sr=None)
#     # wave = wave[::3]
#     mfcc = speechpy.feature.mfcc(audio, sampling_frequency=16000, num_cepstral=20)
#     mean = np.mean(np.ravel(mfcc))
#     std = np.std(np.ravel(mfcc))
#     if std != 0:
#         mfcc = mfcc - mean
#         mfcc = mfcc / std
#     return mfcc

def log_specgram(audio, sr=16000, window_size=20,
                 step_size=10, eps=1e-10):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=52, hop_length=int(0.010*sr), n_fft=int(0.025*sr))
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc_delta)
    # ALthough a test showed not much difference, by eye, it seems rescaling each is better.
    # rescale each matrix
    res = np.array([rescale(mfcc[1:]), rescale(mfcc_delta[1:]), rescale(mfcc_delta2[1:])])
    res = np.swapaxes(res, 2, 0)
    return res

def rescale(m):
    #rescale by global max of absolute values
    offset = m.min()
    scale = m.max()-m.min()
    return (m-offset)/scale

# def log_specgram(audio, sample_rate=16000, window_size=20,
#                  step_size=10, eps=1e-10):
#     nperseg = int(round(window_size * sample_rate / 1e3))
#     noverlap = int(round(step_size * sample_rate / 1e3))
#     spec = librosa.feature.melspectrogram(audio, sr=sample_rate, n_mels=128)
#     spec = librosa.power_to_db(spec, ref=np.max)
#
#     mean = np.mean(np.ravel(spec))
#     std = np.std(np.ravel(spec))
#     if std != 0:
#         spec = spec - mean
#         spec = spec / std
#     return spec


# def log_specgram(data, sr=16000):
#     data = librosa.feature.melspectrogram(data, sr=sr, n_mels=40, hop_length=160, n_fft=480, fmin=20, fmax=4000)
#     data[data > 0] = np.log(data[data > 0])
#     # data = [np.matmul(librosa.filters.dct(40,40), x) for x in np.split(data, data.shape[1], axis=1)]
#     # data = np.array(data, order="F").squeeze(2).astype(np.float32)
#     mean = np.mean(np.ravel(data))
#     std = np.std(np.ravel(data))
#     if std != 0:
#         data = data - mean
#         data = data / std
#
    # return data

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
    wav = signal.resample(wav, 8000)
    mean = np.mean(np.ravel(wav))
    std = np.std(np.ravel(wav))
    if std != 0:
        wav = wav - mean
        wav = wav / std
    return wav

def random_onoff():                # randomly turns on or off
    return bool(random.getrandbits(1))

def augment_data(y, sr, noises, allow_speedandpitch = True, allow_pitch = True,
    allow_speed = True, allow_dyn = True, allow_noise = True, allow_timeshift = True, tab=""):
    length = y.shape[0]
    y_mod = y

    # add noise
    if (allow_noise) and random_onoff():
        # noise_amp = 0.005*np.random.uniform()*np.amax(y)
        # if random_onoff():
        #     y_mod +=  noise_amp * np.random.normal(size=length)
        # else:
        #     y_mod +=  noise_amp * np.random.normal(size=length)
        noise = noises[random.randint(0, len(noises) - 1)]
        scale = np.random.uniform(low=0, high=0.4, size=1)
        y_mod = (1 - scale) * y_mod + (noise * scale)

    # change speed and pitch together
    if (allow_speedandpitch) and random_onoff():
        length_change = np.random.uniform(low=0.9,high=1.1)
        speed_fac = 1.0  / length_change
        tmp = np.interp(np.arange(0,len(y),speed_fac),np.arange(0,len(y)),y)
        #tmp = resample(y,int(length*lengt_fac))    # signal.resample is too slow
        minlen = min( y.shape[0], tmp.shape[0])     # keep same length as original;
        y_mod *= 0                                    # pad with zeros
        y_mod[0:minlen] = tmp[0:minlen]

    # change pitch (w/o speed)
    # if (allow_pitch) and random_onoff():
    #     bins_per_octave = 24        # pitch increments are quarter-steps
    #     pitch_pm = 4                                # +/- this many quarter steps
    #     pitch_change =  pitch_pm * 2*(np.random.uniform()-0.5)
    #     y_mod = librosa.effects.pitch_shift(y, sr, n_steps=pitch_change, bins_per_octave=bins_per_octave)

    # change speed (w/o pitch),
    # if (allow_speed) and random_onoff():
    #     speed_change = np.random.uniform(low=0.9,high=1.1)
    #     tmp = librosa.effects.time_stretch(y_mod, speed_change)
    #     minlen = min( y.shape[0], tmp.shape[0])        # keep same length as original;
    #     y_mod *= 0                                    # pad with zeros
    #     y_mod[0:minlen] = tmp[0:minlen]

    # change dynamic range
    if (allow_dyn) and random_onoff():
        dyn_change = np.random.uniform(low=0.5,high=1.1)  # change amplitude
        y_mod = y_mod * dyn_change

    # shift in time forwards or backwards
    if (allow_timeshift) and random_onoff():
        timeshift_fac = 0.2 *2*(np.random.uniform()-0.5)  # up to 20% of length
        start = int(length * timeshift_fac)
        if (start > 0):
            y_mod = np.pad(y_mod,(start,0),mode='constant')[0:y_mod.shape[0]]
        else:
            y_mod = np.pad(y_mod,(0,-start),mode='constant')[0:y_mod.shape[0]]

    return y_mod
