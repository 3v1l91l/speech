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
import random
# from generator import batch_generator
from lib import get_path_label_df, prepare_data, log_specgram, get_specgrams, get_specgrams_augment_unknown, get_specgrams_augment_known
import librosa

L = 16000
legal_labels = 'yes no up down left right on off stop go silence unknown'.split()
root_path = r'..'
out_path = r'.'
model_path = r'.'
train_data_path = os.path.join(root_path, 'input', 'train', 'audio')
test_data_path = os.path.join(root_path, 'input', 'test', 'audio')
background_noise_paths = glob(os.path.join(train_data_path, r'_background_noise_/*' + '.wav'))
silence_paths = glob(os.path.join(train_data_path, r'silence/*' + '.wav'))
new_sample_rate = 8000

import threading

class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):  # Py3
        with self.lock:
            return next(self.it)

def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g

@threadsafe_generator
def batch_generator(X, y, y_label, silences, unknowns, batch_size=16):
    '''
    Return batch of random spectrograms and corresponding labels
    '''

    while True:
        idx = np.random.randint(0, X.shape[0], batch_size)
        im = X[idx]
        label = y[idx]
        cur_legal_labels = [not x == 'unknown' for x in y_label[idx]]
        cur_not_legal_labels = [not x for x in cur_legal_labels]
        if any(cur_not_legal_labels):  # augment unknowns
            im[np.where(cur_not_legal_labels)] = get_specgrams_augment_unknown(im[cur_not_legal_labels], silences, unknowns)
            # fpath = background_noise_paths[np.random.randint(0, len(background_noise_paths) - 1)]
            # sample_rate, noise = wavfile.read(fpath)
            # beg = np.random.randint(0, len(noise) - L)
            # scale = np.random.uniform(low=0, high=0.5, size=1)
            # noise_sample = noise[beg: beg + L]
            # wav = im[i]
            # im[i] = (1 - scale) * wav + (noise_sample * scale)

        if any(cur_legal_labels):
            im[np.where(cur_legal_labels)] = get_specgrams_augment_known(im[cur_legal_labels], silences, unknowns)
            # fpath = background_noise_paths[np.random.randint(0, len(background_noise_paths) - 1)]
            # sample_rate, noise = wavfile.read(fpath)
            # beg = np.random.randint(0, len(noise) - L)
            # scale = np.random.uniform(low=0, high=0.5, size=1)
            # noise_sample = noise[beg: beg + L]
            # wav = im[i]
            # im[i] = (1 - scale) * wav + (noise_sample * scale)
            # specgram = get_specgrams_augment(im)

        yield np.stack(im), label

def get_specgram_labels(zip):
    x, y = [], []
    label, fname = zip
    sample_rate, orig_samples = wavfile.read(os.path.join(train_data_path, label, fname))
    samples = pad_audio(orig_samples)
    n_samples = chop_audio(samples, label)
    for samples in n_samples:
        resampled = signal.resample(samples, int(new_sample_rate / sample_rate * samples.shape[0]))
        specgram = log_specgram(resampled, sample_rate=new_sample_rate)

        x.append(specgram)
        y.append(label)
    return (x, y)

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
    # input_shape = (99, 81, 1)
    input_shape = (99, 161, 1)

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
    # x_train, y_train = get_x_y(False)
    # y_train = label_transform(y_train)

    train = prepare_data(get_path_label_df('../input/train/audio/'))
    valid = prepare_data(get_path_label_df('../input/train/valid/'))


    len_train = len(train.word.values)
    temp = label_transform(np.concatenate((train.word.values, valid.word.values)))
    y_train, y_valid = temp[:len_train], temp[len_train:]

    label_index = y_train.columns.values
    y_train = np.array(y_train.values)
    y_valid = np.array(y_valid.values)

    model = get_model()
    model_checkpoint = ModelCheckpoint('model.model', monitor='val_acc', save_best_only=True, save_weights_only=False,
                                       verbose=1)

    silences = []
    silence_paths = train.path[train.word == 'silence']
    for p in silence_paths.iloc[np.random.randint(0,len(silence_paths), 100)]:
        wav, s = librosa.load(p)
        if wav.size < L:
            wav = np.pad(wav, (L - wav.size, 0), mode='constant')
        else:
            wav = wav[0:L]
        silences.append(wav)

    unknowns = []
    unknown_paths = train.path[train.word == 'unknown']
    for p in unknown_paths.iloc[np.random.randint(0,len(unknown_paths), 100)]:
        wav, s = librosa.load(p)
        if wav.size < L:
            wav = np.pad(wav, (L - wav.size, 0), mode='constant')
        else:
            wav = wav[0:L]
        unknowns.append(wav)

    batch_size = 128
    train_gen = batch_generator(train.path.values, y_train, train.word.values, silences, unknowns, batch_size=batch_size)
    valid_gen = batch_generator(valid.path.values, y_valid, valid.word.values, silences, unknowns, batch_size=batch_size)

    model.fit_generator(
        generator=train_gen,
        epochs=1,
        steps_per_epoch=len(y_train) // batch_size,
        validation_data=valid_gen,
        validation_steps=len(y_valid) // batch_size,
        callbacks=[
            model_checkpoint
        ], workers=4, verbose=1)

    del train, valid, y_train, y_valid
    gc.collect()

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
            # resampled = signal.resample(samples, int(new_sample_rate / rate * samples.shape[0]))
            specgram = log_specgram(samples)
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