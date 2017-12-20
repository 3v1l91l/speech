import numpy as np
from lib import *
import threading
import math
from multiprocessing import Pool
import os
from keras.callbacks import Callback

legal_labels = 'yes no up down left right on off stop go silence unknown'.split()
legal_labels_without_unknown = 'yes no up down left right on off stop go silence'.split()
legal_labels_without_unknown_and_silence = 'yes no up down left right on off stop go silence'.split()
legal_labels_without_unknown_can_be_flipped = [x for x in legal_labels_without_unknown_and_silence if x[::-1] not in legal_labels_without_unknown_and_silence]

from keras import backend as K
class LearningRateTracker(Callback):
    def on_epoch_end(self, epoch, logs=None):
        lr = self.model.optimizer.lr
        decay = self.model.optimizer.decay
        iterations = self.model.optimizer.iterations
        lr_with_decay = lr / (1. + decay * K.cast(iterations, K.dtype(decay)))
        print("Learning rate: {}".format(K.eval(lr_with_decay)))

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
def batch_generator_paths(X_paths, y, y_label, silences, batch_size=128):
    while True:
        known_ix = np.random.choice(y_label.index, size=batch_size)
        X = list(map(load_wav_by_path, X_paths[known_ix]))

        specgrams = get_specgrams_augment_known(X, silences)
        res_labels = y[known_ix]

        yield np.stack(specgrams), res_labels

@threadsafe_generator
def batch_generator_paths_old(validate, X_paths, y, y_label, silences, batch_size=16):
    while True:
        # Try to represent classes distribution
        batch_size_unknown = math.ceil(0.15 * batch_size)
        batch_size_unknown_flip_known = 0
        if validate:
            batch_size_unknown_flip_known = math.ceil(0.15 * batch_size)
        batch_size_silence = math.ceil(0.1 * batch_size)
        batch_size_known = batch_size - batch_size_unknown - batch_size_silence - batch_size_unknown_flip_known
        unknown_ix = np.random.choice(y_label[y_label == 'unknown'].index, size=batch_size_unknown)
        unknown_flip_known_ix = np.random.choice(y_label[y_label.isin(legal_labels_without_unknown_can_be_flipped)].index, size=batch_size_unknown_flip_known)
        silence_ix = np.random.choice(y_label[y_label == 'silence'].index, size=batch_size_silence)
        known_ix = np.random.choice(y_label[(y_label != 'unknown') & (y_label != 'silence')].index, size=batch_size_known)
        all_unknown_ix = np.concatenate((unknown_ix,unknown_flip_known_ix))
        X = list(map(load_wav_by_path, np.concatenate((X_paths[all_unknown_ix],X_paths[silence_ix],X_paths[known_ix]))))

        specgrams = []
        res_labels = []

        specgrams.extend(get_specgrams_augment_unknown_flip(X[:len(all_unknown_ix)], len(unknown_ix) + np.array(range(len(unknown_flip_known_ix))), silences))
        res_labels.extend(y[all_unknown_ix])

        specgrams.extend(get_specgrams_augment_silence(X[len(all_unknown_ix):len(all_unknown_ix) + len(silence_ix)], silences))
        res_labels.extend(y[silence_ix])

        specgrams.extend(get_specgrams_augment_known(X[len(all_unknown_ix)+len(silence_ix):], silences))
        res_labels.extend(y[known_ix])

        res_labels = np.concatenate((y[all_unknown_ix],y[silence_ix],y[known_ix]))
        yield np.stack(specgrams), res_labels


@threadsafe_generator
def batch_generator_silence_paths(X_paths, y, y_label, silences, batch_size=128):
    while True:
        # Try to represent classes distribution
        batch_size_unknown = math.ceil(0.5 * batch_size)
        batch_size_silence = batch_size - batch_size_unknown
        unknown_ix = np.random.choice(y_label[y_label == 'unknown'].index, size=batch_size_unknown)
        silence_ix = np.random.choice(y_label[y_label == 'silence'].index, size=batch_size_silence)
        X = list(map(load_wav_by_path, np.concatenate((X_paths[unknown_ix], X_paths[silence_ix]))))

        specgrams = []
        res_labels = []
        specgrams.extend(get_specgrams_augment_unknown(X[:len(unknown_ix)], silences))
        res_labels.extend(y[unknown_ix])

        specgrams.extend(get_specgrams_augment_silence(X[len(unknown_ix):], silences))
        res_labels.extend(y[silence_ix])

        res_labels = np.concatenate((y[unknown_ix], y[silence_ix]))
        yield np.stack(specgrams), res_labels

@threadsafe_generator
def batch_generator_unknown_paths(X_paths, y, y_label, silences, batch_size=128):
    while True:
        # Try to represent classes distribution
        batch_size_unknown = math.ceil(0.5 * batch_size)
        batch_size_silence = batch_size - batch_size_unknown
        unknown_ix = np.random.choice(y_label[y_label == 'unknown'].index, size=batch_size_unknown)
        known_ix = np.random.choice(y_label[y_label == 'known'].index, size=batch_size_silence)
        X = list(map(load_wav_by_path, np.concatenate((X_paths[unknown_ix], X_paths[known_ix]))))

        specgrams = []
        res_labels = []
        specgrams.extend(get_specgrams_augment_unknown(X[:len(unknown_ix)], silences))
        res_labels.extend(y[unknown_ix])

        specgrams.extend(get_specgrams_augment_known(X[len(unknown_ix):], silences))
        res_labels.extend(y[known_ix])

        res_labels = np.concatenate((y[unknown_ix], y[known_ix]))
        yield np.stack(specgrams), res_labels

def test_data_generator(fpaths, batch=16):
    i = 0
    for path in fpaths:
        if i == 0:
            imgs = []
            fnames = []
        samples = load_wav_by_path(path)
        specgram = log_specgram(samples)
        imgs.append(specgram)
        fnames.append(path.split(os.sep)[-1])
        i += 1
        if i == batch:
            i = 0
            imgs = np.array(imgs)[..., np.newaxis]
            # imgs = np.array(imgs)
            yield fnames, imgs
    if i < batch:
        imgs = np.array(imgs)[..., np.newaxis]
        # imgs = np.array(imgs)
        yield fnames, imgs
    raise StopIteration()

def valid_data_generator(fpaths, batch=16):
    imgs = []
    fnames = []
    i = 0
    for path in fpaths:
        samples = load_wav_by_path(path)
        specgram = log_specgram(samples)
        imgs.append(specgram)
        folder = path.split(os.sep)[-2]
        if folder not in legal_labels_without_unknown:
            folder = 'unknown'
        fnames.append(folder)
        i += 1
        if i == batch:
            i = 0
            imgs = np.array(imgs)[..., np.newaxis]
            # imgs = np.array(imgs)
            yield fnames, imgs
            imgs = []
            fnames = []
    if (i < batch) and (len(imgs)>0):
        imgs = np.array(imgs)[..., np.newaxis]
        # imgs = np.array(imgs)
        yield fnames, imgs
    raise StopIteration()