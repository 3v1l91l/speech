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
background_noise_paths = glob(os.path.join(train_data_path, r'_background_noise_', '*wav'))
silence_paths = glob(os.path.join(train_data_path, 'silence', '*wav'))

legal_labels = 'yes no up down left right on off stop go silence unknown'.split()

def get_path_label_df(path, pattern='**' + os.sep + '*.wav'):
    datadir = Path(path)
    files = [(str(f), f.parts[-2]) for f in datadir.glob(pattern) if f]
    df = pd.DataFrame(files, columns=['path', 'word'])

    return df

def prepare_data(df):
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

# def get_specgrams_silence(wavs):
#     log_specgrams = [None] * len(wavs)
#     fs = 16000
#     for i in range(len(wavs)):
#         log_specgrams[i] = log_specgram(wavs[i], fs)[..., np.newaxis]
#     return log_specgrams

def get_specgrams_augment_known(wavs, silences, unknowns):
    len_paths = len(wavs)
    log_specgrams = [None]*len_paths
    fs = 16000
    for i in range(len(wavs)):
        wav = augment_data(wavs[i], fs, silences, unknowns)
        log_specgrams[i] = log_specgram(wav, fs)[..., np.newaxis]
    return log_specgrams

def get_specgrams_augment_known_valid(wavs, silences, unknowns):
    len_paths = len(wavs)
    log_specgrams = [None]*len_paths
    fs = 16000
    for i in range(len(wavs)):
        wav = augment_data_valid(wavs[i], fs, silences, unknowns)
        log_specgrams[i] = log_specgram(wav, fs)[..., np.newaxis]
    return log_specgrams

def get_specgrams_augment_unknown(wavs, silences, unknowns):
    if len(wavs) == 0:
        print('err')
    len_paths = len(wavs)
    log_specgrams = [None]*len_paths
    fs = 16000
    duration = 1
    for i in range(len(wavs)):
        wav = augment_unknown(wavs[i], fs, silences, unknowns)
        log_specgrams[i] = log_specgram(wav, fs)[..., np.newaxis]
    return log_specgrams

def get_specgrams_augment_silence(wavs, silences, unknowns):
    if len(wavs) == 0:
        print('err')
    len_paths = len(wavs)
    log_specgrams = [None]*len_paths
    fs = 16000
    duration = 1
    for i in range(len(wavs)):
        wav = augment_silence(wavs[i], fs, silences, unknowns)
        log_specgrams[i] = log_specgram(wav, fs)[..., np.newaxis]
    return log_specgrams

def log_specgram(audio, sr=16000):
    n_mfcc = 40
    # hop_length = int(0.02 * sr)
    # n_fft = int(0.04 * sr)
    window_size_ms = 30.0
    window_stride_ms = 10.0
    window_size_samples = int(sr * window_size_ms / 1000)
    window_stride_samples = int(sr * window_stride_ms / 1000)

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc, hop_length=window_stride_samples, n_fft=window_size_samples)
    # mean = np.mean(np.ravel(mfcc))
    # std = np.std(np.ravel(mfcc))
    # if std != 0:
    #     mfcc = mfcc - mean
    #     mfcc = mfcc / std
    return mfcc

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

    # loudest_section_dur = 0.4
    # loudest_section_samples = int(loudest_section_dur * L)
    # max_loudness = np.sum(wav[0:loudest_section_samples])
    # max_loudness_ix = 0
    # for i in range(0, L-loudest_section_samples):
    #     if np.sum(wav[i:i+loudest_section_samples]) > max_loudness:
    #         max_loudness_ix = i
    # wav = np.concatenate(wav[max_loudness_ix:max_loudness_ix+loudest_section_samples], wav[0:max_loudness_ix],
    #                      wav[max_loudness_ix+loudest_section_samples:])

    # wav = signal.resample(wav, 8000)
    # mean = np.mean(np.ravel(wav))
    # std = np.std(np.ravel(wav))
    # if std != 0:
    #     wav = wav - mean
    #     wav = wav / std
    return wav

def random_onoff():                # randomly turns on or off
    return bool(random.getrandbits(1))

def augment_unknown(y, sr, noises, unknowns):
    y_mod = y
    if random_onoff():
        y_mod = np.flip(y_mod, axis=0)

    augment_data(y_mod,sr,noises,unknowns)
    return y_mod

def augment_silence(y, sr, noises, unknowns, allow_speedandpitch = True, allow_pitch = True,
    allow_speed = True, allow_dyn = True, allow_noise = True, allow_timeshift = True, tab=""):
    y_mod = y
    multiplier = np.random.uniform()
    y_mod = y_mod * multiplier
    return y_mod

def augment_data_valid(y, sr, noises, unknowns, allow_speedandpitch = True, allow_pitch = True,
    allow_speed = True, allow_dyn = True, allow_noise = True, allow_timeshift = True, tab=""):
    length = y.shape[0]
    y_mod = y

    if random_onoff():
        timeshift_fac = 0.15 *2*(np.random.uniform()-0.5)
        start = int(length * timeshift_fac)
        if (start > 0):
            y_mod = np.pad(y_mod,(start,0),mode='constant')[0:y_mod.shape[0]]
        else:
            y_mod = np.pad(y_mod,(0,-start),mode='constant')[0:y_mod.shape[0]]
    return y_mod


def augment_data(y, sr, noises, unknowns, allow_speedandpitch = True, allow_pitch = True,
    allow_speed = True, allow_dyn = True, allow_noise = True, allow_timeshift = True, tab=""):
    length = y.shape[0]
    y_mod = y

    # add noise


    # # shift in time forwards or backwards
    # # if  random_onoff():
    # shift_samples_num = int(((-1) ** random.randrange(2)) * sr * random.uniform(0, 0.15))
    # y_mod = np.roll(y_mod, shift_samples_num)

    if random_onoff():
        timeshift_fac = 0.15 *2*(np.random.uniform()-0.5)
        start = int(length * timeshift_fac)
        if (start > 0):
            y_mod = np.pad(y_mod,(start,0),mode='constant')[0:y_mod.shape[0]]
        else:
            y_mod = np.pad(y_mod,(0,-start),mode='constant')[0:y_mod.shape[0]]

    # change speed and pitch together
    if random_onoff():
        length_change = np.random.uniform(low=0.7,high=1.3)
        speed_fac = 1.0  / length_change
        tmp = np.interp(np.arange(0,len(y),speed_fac),np.arange(0,len(y)),y)
        #tmp = resample(y,int(length*lengt_fac))    # signal.resample is too slow
        minlen = min( y.shape[0], tmp.shape[0])     # keep same length as original;
        y_mod *= 0                                    # pad with zeros
        y_mod[0:minlen] = tmp[0:minlen]

    if (allow_noise) and random_onoff():
        # noise_amp = 0.005*np.random.uniform()*np.amax(y)
        # if random_onoff():
        #     y_mod +=  noise_amp * np.random.normal(size=length)
        # else:
        #     y_mod +=  noise_amp * np.random.normal(size=length)
        noise = noises[random.randint(0, len(noises) - 1)]
        scale = np.random.uniform(low=0, high=0.2, size=1)
        if np.max(noise) > 0 :
            y_mod = (1 - scale) * y_mod + (noise * (np.max(y_mod)/ np.max(noise)) * scale)

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
    # if (allow_dyn) and random_onoff():
    #     dyn_change = np.random.uniform(low=0.5,high=1.1)  # change amplitude
    #     y_mod = y_mod * dyn_change

    return y_mod
