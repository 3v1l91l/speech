from pathlib import Path
from scipy.io import wavfile
import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt

datadir = Path('../input/train/audio/off/')

files = [str(f) for f in datadir.glob('**/*.wav') if f]

fs = 16000
duration = 1
wavs = [wavfile.read(x)[1] for x in files[:5]]
data = []
for wav in wavs:
    if wav.size < fs:
        d = np.pad(wav, (fs * duration - wav.size, 0), mode='constant')
    else:
        d = wav[0:fs * duration]
    data.append(d)


window_size=20
step_size=10
eps=1e-10
sample_rate=16000
nperseg = int(round(window_size * sample_rate / 1e3))
noverlap = int(round(step_size * sample_rate / 1e3))
freqs, times, spec = signal.spectrogram(data[1], fs=sample_rate, window='hann',
    nperseg=nperseg,
    noverlap=noverlap,
    detrend=False)



plt.imshow(np.log(spec.T.astype(np.float32) + eps))
plt.show()


# n = 0.005/4
# no = n / 2
# specgram = [signal.spectrogram(d, fs=fs)[2] for d in data]
# sshape = specgram[0].shape
# print(sshape)
# plt.imshow(specgram[1])
# plt.show()
# specgram = [s.reshape(shape[0], shape[1], -1) for s in specgram]