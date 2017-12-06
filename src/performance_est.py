import os
from glob import glob
from scipy.io import wavfile
import time
train_data_path = os.path.join('..', 'input', 'train', 'audio', 'yes')
import librosa
from scikits.talkbox.features import mfcc

sr = 16000
paths = glob(os.path.join(train_data_path, '*wav'))
start = time.time()
for path in paths:
    _, wav = wavfile.read(path)
    mfcc(wav, sampling_frequency=sr, num_cepstral=40, nwin=int(0.02 * sr), nfft=int(0.04 * sr))
end = time.time()
print('speechpy elapsed {}'.format(end-start))

start = time.time()
for path in paths:
    _, wav = wavfile.read(path)
    librosa.feature.mfcc(y=wav, sr=sr, n_mfcc=40, hop_length=int(0.02 * sr), n_fft=int(0.04 * sr))
end = time.time()
print('librosa elapsed {}'.format(end-start))