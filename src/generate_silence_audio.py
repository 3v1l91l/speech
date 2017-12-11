from lib import *
from scipy.io import wavfile
import numpy as np
import os

train_dir = os.path.join('..', 'input', 'train')
back_noise_dir = os.path.join('..', 'input', 'train')
# silence_audio_dir = os.path.join(train_dir, 'audio', 'silence')
# silence_audio_dir = os.path.join(train_dir, 'valid','silence')
silence_audio_dir = os.path.join(train_dir, 'test','silence')

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
    # multiplier = np.random.rand()*0.00001
    multiplier = 1
    gen_wav = multiplier * wavs[i][sec_start * fs : (sec_start+1) * fs]
    wavfile.write(os.path.join(silence_audio_dir, 'noise_' + str(count) + '.wav'), fs, gen_wav)
    count += 1