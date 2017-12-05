import speechpy
import os
from glob import glob
from scipy.io import wavfile
import time
train_data_path = os.path.join('..', 'input', 'train', 'audio', 'yes')
import librosa

paths = glob(os.path.join(train_data_path, '*wav'))
start = time.time()
for path in paths:
    _, wav = wavfile.read(path)
    speechpy.feature.mfcc(wav, sampling_frequency=16000, num_cepstral=20)
end = time.time()
print('speechpy elapsed {}'.format(end-start))

start = time.time()
for path in paths:
    _, wav = wavfile.read(path)
    librosa.feature.mfcc(wav, sr=16000)
end = time.time()
print('librosa elapsed {}'.format(end-start))