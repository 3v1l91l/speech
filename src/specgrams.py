from pathlib import Path
from scipy.io import wavfile
import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
from lib import get_specgrams

datadir = Path('../input/train/audio/off/')

files = [str(f) for f in datadir.glob('**/*.wav') if f]

specs = get_specgrams(files)

# print(specs[0].shape)
# plt.imshow(specs[0].reshape(specs[0].shape[0], specs[0].shape[1]))
# plt.show()


# n = 0.005/4
# no = n / 2
# specgram = [signal.spectrogram(d, fs=fs)[2] for d in data]
# sshape = specgram[0].shape
# print(sshape)
# plt.imshow(specgram[1])
# plt.show()
# specgram = [s.reshape(shape[0], shape[1], -1) for s in specgram]