from lib import *
from scipy.io import wavfile
import numpy as np
import os
from numpy.fft import rfft, irfft

train_dir = os.path.join('..', 'input', 'train')
back_noise_dir = os.path.join('..', 'input', 'train')
silence_audio_dirs = [os.path.join(train_dir, 'test','silence'),
                      os.path.join(train_dir, 'valid','silence'),
                      os.path.join(train_dir, 'audio', 'silence')]
state = np.random.RandomState()

def to_16bit_audio(wav):
    return np.int16(wav / np.max(np.abs(wav)) * 32767)

def generate_and_save_silences(silence_audio_dir):
    if not os.path.exists(silence_audio_dir):
        os.makedirs(silence_audio_dir)

    df = get_path_label_df(os.path.join(back_noise_dir, '_background_noise_'))
    wavs = [wavfile.read(x)[1] for x in df.path]
    gen_silence_files_num = 2000
    idx = np.random.randint(0, len(wavs)-1, gen_silence_files_num)
    fs = 16000
    gen_wavs = []
    count = 0
    for i in idx:
        sec_length = (wavs[i].shape[0] // fs) - 1
        sec_start = np.random.randint(0, sec_length)
        gen_wav = wavs[i][sec_start * fs : (sec_start+1) * fs]
        wavfile.write(os.path.join(silence_audio_dir, 'noise_' + str(count) + '.wav'), fs, gen_wav)
        count += 1

    for i in range(200):
        gen_wav = np.array([0] * fs, dtype=np.int16)
        wavfile.write(os.path.join(silence_audio_dir, 'noise_' + str(count) + '.wav'), fs, gen_wav)
        count += 1

    for i in range(200):
        gen_wav = np.random.choice([-16000, 0, 16000], fs)
        wavfile.write(os.path.join(silence_audio_dir, 'noise_' + str(count) + '.wav'), fs, gen_wav)
        count += 1

    # blue_noise
    for i in range(200):
        X = state.randn(fs // 2 + 1) + 1j * state.randn(fs // 2 + 1)
        S = np.sqrt(np.arange(len(X)))  # Filter
        gen_wav = (irfft(X * S)).real
        gen_wav = to_16bit_audio(gen_wav)
        wavfile.write(os.path.join(silence_audio_dir, 'noise_' + str(count) + '.wav'), fs, gen_wav)
        count += 1

    # brown_noise
    for i in range(200):
        X = state.randn(fs // 2 + 1) + 1j * state.randn(fs // 2 + 1)
        S = (np.arange(len(X)) + 1)  # Filter
        gen_wav = (irfft(X / S)).real
        gen_wav = to_16bit_audio(gen_wav)
        wavfile.write(os.path.join(silence_audio_dir, 'noise_' + str(count) + '.wav'), fs, gen_wav)
        count += 1

    # violet_noise
    for i in range(200):
        X = state.randn(fs // 2 + 1) + 1j * state.randn(fs // 2 + 1)
        S = (np.arange(len(X)))  # Filter
        gen_wav = (irfft(X * S)).real
        gen_wav = to_16bit_audio(gen_wav)
        wavfile.write(os.path.join(silence_audio_dir, 'noise_' + str(count) + '.wav'), fs, gen_wav)
        count += 1

for silence_audio_dir in silence_audio_dirs:
    generate_and_save_silences(silence_audio_dir)
