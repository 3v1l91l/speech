import os
import numpy as np
from scipy.fftpack import fft
from scipy.io import wavfile
from scipy import signal
from glob import glob
import re
import pandas as pd
import gc
from scipy.io import wavfile
import pickle
from keras import optimizers, losses, activations, models
from keras.layers import Convolution2D, Dense, Input, Flatten, Dropout, MaxPooling2D, BatchNormalization
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from keras.models import load_model
from tqdm import tqdm
from multiprocessing import Pool
import time
import matplotlib.pyplot as plt
import pyrubberband as pyrb
from librosa import effects
import random

L = 16000
new_sample_rate = 8000
legal_labels = 'yes no up down left right on off stop go silence unknown'.split()
root_path = r'..'
out_path = r'.'
model_path = r'.'
train_data_path = os.path.join(root_path, 'input', 'train', 'audio')
test_data_path = os.path.join(root_path, 'input', 'test', 'audio')
background_noise_paths = glob(os.path.join(train_data_path, r'_background_noise_/*' + '.wav'))


def get_specgram_labels(zip):
    x, y = [], []
    label, fname = zip
    sample_rate, orig_samples = wavfile.read(os.path.join(train_data_path, label, fname))
    samples = pad_audio(orig_samples)
    n_samples = chop_audio(samples, label)
    for samples in n_samples:
        resampled = signal.resample(samples, int(new_sample_rate / sample_rate * samples.shape[0]))
        _, _, specgram = log_specgram(resampled, sample_rate=new_sample_rate)

        x.append(specgram)
        y.append(label)
    return (x, y)

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
    return freqs, times, np.log(spec.T.astype(np.float32) + eps)

def list_wavs_fname(dirpath, ext='wav'):
    print(dirpath)
    fpaths = glob(os.path.join(dirpath, r'*/*' + ext))
    pat = r'.+/(\w+)/.+' + ext + '$'
    labels = []
    for fpath in fpaths:
        r = re.match(pat, fpath)
        if r:
            labels.append(r.group(1))
    pat = r'.+/(.+' + ext + ')$'
    fnames = []
    for fpath in fpaths:
        r = re.match(pat, fpath)
        if r:
            fnames.append(r.group(1))
    return labels, fnames

def pad_audio(samples):
    if len(samples) >= L: return samples
    else: return np.pad(samples, pad_width=(L - len(samples), 0), mode='constant', constant_values=(0, 0))

def chop_audio(samples, label, L=16000):
    if label == '_background_noise_':   # chop noise to be 'silences'
        num_silences_per_record = 500
        for i in range(num_silences_per_record):
            scale = np.random.uniform(low=0, high=1, size=1)
            beg = np.random.randint(0, len(samples) - L)
            yield samples[beg: beg + L] * scale
    elif label not in legal_labels:     # augment unknowns
        fpath = background_noise_paths[np.random.randint(0, len(background_noise_paths)-1)]
        sample_rate, noise = wavfile.read(fpath)
        num_augmented = 3
        linspace = [int(x) for x in np.linspace(0, len(samples), num_augmented+1)]
        for i in range(num_augmented):
            beg = np.random.randint(0, len(noise) - L)
            scale = np.random.uniform(low=0, high=0.5, size=1)
            noise_sample = noise[beg: beg + L]
            wav = np.concatenate((samples[linspace[i]:], samples[:linspace[i]]))
            if bool(random.getrandbits(1)):
                wav = wav * 0.5 + linspace[-1] * 0.5
            yield (1 - scale) * wav + (noise_sample * scale)
    else:
        num_augmented = 10
        fpath = background_noise_paths[np.random.randint(0, len(background_noise_paths) - 1)]
        sample_rate, noise = wavfile.read(fpath)
        for i in range(num_augmented):
            scale = np.random.uniform(low=0, high=0.5, size=1)
            beg = np.random.randint(0, len(noise) - L)
            noise_sample = noise[beg: beg + L]
            wav = samples
            yield (1 - scale) * wav + noise_sample * scale


def label_transform(labels):
    nlabels = []
    for label in labels:
        if label == '_background_noise_':
            nlabels.append('silence')
        elif label not in legal_labels:
            nlabels.append('unknown')
        else:
            nlabels.append(label)
    return pd.get_dummies(pd.Series(nlabels))

def get_x_y(data_is_loaded=False):
    if data_is_loaded:
        x_train = np.load('./x_train.npy')
        y_train = np.load('./y_train.npy')
    else:
        labels, fnames = list_wavs_fname(train_data_path)

        print('started')
        start = time.time()
        x_train, y_train = [], []
        with Pool() as p:
            for a, b in p.map(get_specgram_labels, zip(labels, fnames)):
                x_train.extend(a)
                y_train.extend(b)

        end = time.time()
        print('Executed in: {}'.format(end - start))
        x_train = np.array(x_train)
        x_train = x_train.reshape(tuple(list(x_train.shape) + [1]))
        y_train = np.array(y_train)
        # np.save('./x_train', x_train)
        # np.save('./y_train', y_train)
    return x_train, y_train

def get_model():
    input_shape = (99, 81, 1)
    nclass = 12
    inp = Input(shape=input_shape)
    norm_inp = BatchNormalization()(inp)
    img_1 = Convolution2D(8, kernel_size=2, activation=activations.relu)(norm_inp)
    img_1 = Convolution2D(8, kernel_size=2, activation=activations.relu)(img_1)
    img_1 = MaxPooling2D(pool_size=(2, 2))(img_1)
    img_1 = Dropout(rate=0.2)(img_1)
    img_1 = Convolution2D(16, kernel_size=3, activation=activations.relu)(img_1)
    img_1 = Convolution2D(16, kernel_size=3, activation=activations.relu)(img_1)
    img_1 = MaxPooling2D(pool_size=(2, 2))(img_1)
    img_1 = Dropout(rate=0.2)(img_1)
    img_1 = Convolution2D(32, kernel_size=3, activation=activations.relu)(img_1)
    img_1 = MaxPooling2D(pool_size=(2, 2))(img_1)
    img_1 = Dropout(rate=0.2)(img_1)
    img_1 = Flatten()(img_1)

    dense_1 = BatchNormalization()(Dense(128, activation=activations.relu)(img_1))
    dense_1 = BatchNormalization()(Dense(128, activation=activations.relu)(dense_1))
    dense_1 = Dense(nclass, activation=activations.softmax)(dense_1)

    model = models.Model(inputs=inp, outputs=dense_1)
    opt = optimizers.Adam()

    model.compile(optimizer=opt, loss=losses.binary_crossentropy, metrics=['accuracy'])
    return model

def main():
    x_train, y_train = get_x_y(False)
    y_train = label_transform(y_train)
    label_index = y_train.columns.values
    y_train = np.array(y_train.values)
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.1, random_state=2017)

    model = get_model()
    model_checkpoint = ModelCheckpoint('model.model', monitor='val_acc', save_best_only=True, save_weights_only=False,
                                       verbose=1)
    model.fit(x_train, y_train, batch_size=128, validation_data=(x_valid, y_valid), epochs=5, shuffle=True, verbose=1,
              callbacks=[model_checkpoint])
    model = load_model('model.model')

    def test_data_generator(batch=16):
        fpaths = glob(os.path.join(test_data_path, '*wav'))
        i = 0
        for path in fpaths:
            if i == 0:
                imgs = []
                fnames = []
            i += 1
            rate, samples = wavfile.read(path)
            samples = pad_audio(samples)
            resampled = signal.resample(samples, int(new_sample_rate / rate * samples.shape[0]))
            _, _, specgram = log_specgram(resampled, sample_rate=new_sample_rate)
            imgs.append(specgram)
            fnames.append(path.split(r'/')[-1])
            if i == batch:
                i = 0
                imgs = np.array(imgs)
                imgs = imgs.reshape(tuple(list(imgs.shape) + [1]))
                yield fnames, imgs
        if i < batch:
            imgs = np.array(imgs)
            imgs = imgs.reshape(tuple(list(imgs.shape) + [1]))
            yield fnames, imgs
        raise StopIteration()

    del x_train, y_train
    gc.collect()

    index = []
    results = []
    for fnames, imgs in test_data_generator(batch=500):
        predicts = model.predict(imgs)
        predicts = np.argmax(predicts, axis=1)
        predicts = [label_index[p] for p in predicts]
        index.extend(fnames)
        results.extend(predicts)
    df = pd.DataFrame(columns=['fname', 'label'])
    df['fname'] = index
    df['label'] = results
    df.to_csv(os.path.join(out_path, 'sub.csv'), index=False)

if __name__ == "__main__":
    main()