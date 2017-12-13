from lib import *
from scipy.io import wavfile
import numpy as np
import os

train_dir = os.path.join('..', 'input', 'train')
back_noise_dir = os.path.join('..', 'input', 'train')
silence_audio_dirs = [os.path.join(train_dir, 'test','silence'),
                      os.path.join(train_dir, 'valid','silence'),
                      os.path.join(train_dir, 'audio', 'silence')]

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
        # multiplier = np.random.uniform()
        multiplier = np.random.uniform(0.5, 2, 1)
        gen_wav = multiplier * wavs[i][sec_start * fs : (sec_start+1) * fs]
        wavfile.write(os.path.join(silence_audio_dir, 'noise_' + str(count) + '.wav'), fs, gen_wav)
        count += 1

    for i in range(500):
        gen_wav = np.array([0] * fs)
        wavfile.write(os.path.join(silence_audio_dir, 'noise_' + str(count) + '.wav'), fs, gen_wav)
        count += 1

    for i in range(500):
        # multiplier = np.random.uniform(0.1, 3, 1)
        multiplier = 3
        gen_wav = np.random.choice([-1, 0, 1], fs) * multiplier
        wavfile.write(os.path.join(silence_audio_dir, 'noise_' + str(count) + '.wav'), fs, gen_wav)
        count += 1

for silence_audio_dir in silence_audio_dirs:
    generate_and_save_silences(silence_audio_dir)
