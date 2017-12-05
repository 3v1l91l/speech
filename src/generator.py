import numpy as np
from lib import *
import threading
import math
from multiprocessing import Pool

legal_labels = 'yes no up down left right on off stop go silence unknown'.split()
legal_labels_without_unknown = 'yes no up down left right on off stop go silence'.split()

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
def batch_generator(should_augment, X, y, y_label, silences, unknowns, batch_size=16):
    while True:
        # Try to represent classes distribution
        batch_size_unknown = math.ceil(0.1 * batch_size)
        batch_size_silence = math.ceil(0.1 * batch_size)
        batch_size_known = batch_size - batch_size_unknown - batch_size_silence
        unknown_ix = np.random.choice(y_label[y_label == 'unknown'].index, size=batch_size_unknown)
        silence_ix = np.random.choice(y_label[y_label == 'silence'].index, size=batch_size_silence)
        known_ix = np.random.choice(y_label[(y_label != 'unknown') & (y_label != 'silence')].index, size=batch_size_known)

        specgrams = []
        res_labels = []
        if should_augment:
            specgrams.extend(get_specgrams_augment_unknown(X[unknown_ix], silences, unknowns))
            # specgrams.extend(p.map(get_specgrams_aug, X[unknown_ix]))
        else:
            specgrams.extend(get_specgrams(X[unknown_ix]))
        res_labels.extend(y[unknown_ix])

        if should_augment:
            # specgrams.extend(p.map(get_specgrams_aug, X[known_ix]))
            specgrams.extend(get_specgrams_augment_known(X[known_ix], silences, unknowns))
        else:
            specgrams.extend(get_specgrams(X[known_ix]))
        res_labels.extend(y[known_ix])

        specgrams.extend(get_specgrams_augment_unknown(X[silence_ix]))
        res_labels.extend(y[silence_ix])

        yield np.stack(specgrams), np.array(res_labels)

def test_data_generator(fpaths, batch=16):
    i = 0
    for path in fpaths:
        if i == 0:
            imgs = []
            fnames = []
        samples = load_wav_by_path(path)
        specgram = log_specgram(samples)
        imgs.append(specgram)
        fnames.append(path.split(r'/')[-1])
        i += 1
        if i == batch:
            i = 0
            imgs = np.array(imgs)[..., np.newaxis]
            yield fnames, imgs
    if i < batch:
        imgs = np.array(imgs)[..., np.newaxis]
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
        folder = path.split('\\')[-2]
        if folder not in legal_labels_without_unknown:
            folder = 'unknown'
        fnames.append(folder)
        i += 1
        if i == batch:
            i = 0
            imgs = np.array(imgs)[..., np.newaxis]
            yield fnames, imgs
            imgs = []
            fnames = []
    if (i < batch) and (len(imgs)>0):
        imgs = np.array(imgs)[..., np.newaxis]
        yield fnames, imgs
    raise StopIteration()