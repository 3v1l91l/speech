import os
import numpy as np
from scipy import signal
from glob import glob
import re
import pandas as pd
import gc
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, Callback
from keras.models import load_model
from tqdm import tqdm
from multiprocessing import Pool
import time
import random
from lib import *
import scipy.io.wavfile as wavfile
import cProfile
from line_profiler import LineProfiler
import math
from generator import *
import lightgbm as lgb

from model import get_model

L = 16000
legal_labels = 'yes no up down left right on off stop go silence unknown'.split()
root_path = r'..'
out_path = r'.'
model_path = r'.'
train_data_path = os.path.join(root_path, 'input', 'train', 'audio')
valid_data_path = os.path.join(root_path, 'input', 'train', 'valid')
test_data_path = os.path.join(root_path, 'input', 'test', 'audio')
background_noise_paths = glob(os.path.join(train_data_path, r'_background_noise_/*' + '.wav'))
silence_paths = glob(os.path.join(train_data_path, r'silence/*' + '.wav'))

def get_predicts(model, label_index):
    fpaths = glob(os.path.join(test_data_path, '*wav'))
    index = []
    results = []
    batch = 128
    for fnames, imgs in tqdm(test_data_generator(fpaths, batch), total=math.ceil(len(fpaths) / batch)):
        predicts = model.predict(imgs)
        predicts = np.argmax(predicts, axis=1)
        predicts = [label_index[p] for p in predicts]
        index.extend(fnames)
        results.extend(predicts)
    return index, results

def validate(model, label_index, path):
    zz = label_index == 'unknown'
    batch = 128
    all_fpaths = glob(os.path.join(path, '*/*wav'))
    all_folders = next(os.walk(train_data_path))[1]
    all_correct_count = 0
    for f in all_folders:
        correct_count = 0
        fpaths = [fp for fp in all_fpaths if fp.split(r'/')[-2] == f]
        for labels, imgs in valid_data_generator(fpaths, batch):
            predicts = model.predict(imgs)
            # predicts = [zz if np.max(p) < 0.3 else p for p in predicts ]
            predicts = np.argmax(predicts, axis=1)
            predicts = [label_index[p] for p in predicts]
            correct_count += np.sum(np.array(predicts) == np.array(labels))
            all_correct_count += np.sum(np.array(predicts) == np.array(labels))
        print('Correct predicted for label {}: {}%'.format(f, correct_count / len(fpaths)))
    print('=======')
    print('Overall correctly predicted: {}%'.format(100 * all_correct_count / len(all_fpaths)))

def main():
    train = prepare_data(get_path_label_df('../input/train/audio/'))
    valid = prepare_data(get_path_label_df('../input/train/valid/'))

    len_train = len(train.word.values)
    len_valid = len(valid.word.values)
    temp = label_transform(np.concatenate((train.word.values, valid.word.values)))
    y_train, y_valid = temp[:len_train], temp[len_train:]

    label_index = y_train.columns.values
    y_train = np.array(y_train.values)
    y_valid = np.array(y_valid.values)

    # model = load_model('model.model')

    model = get_model()
    model.load_weights('model.model')
    model_checkpoint = ModelCheckpoint('model.model', monitor='val_loss', save_best_only=True, save_weights_only=False,
                                       verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=4, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=1, verbose=1)
    # lr_tracker = LearningRateTracker()

    start = time.time()
    pool = Pool()
    silence_paths = train.path[train.word == 'silence']
    rand_silence_paths = silence_paths.iloc[np.random.randint(0,len(silence_paths), 50)]
    silences = np.array(list(pool.imap(load_wav_by_path, rand_silence_paths)))

    unknown_paths = train.path[train.word == 'unknown']
    rand_unknown_paths = unknown_paths.iloc[np.random.randint(0,len(unknown_paths), 50)]
    unknowns = np.array(list(pool.imap(load_wav_by_path, rand_unknown_paths)))

    train_wavs = np.array(list(pool.imap(load_wav_by_path, train.path.values)))
    valid_wavs = np.array(list(pool.imap(load_wav_by_path, valid.path.values)))
    end = time.time()
    print('read files in {}'.format(end-start))

    batch_size = 128
    train_gen = batch_generator(True, train_wavs, y_train, train.word, silences, unknowns, batch_size=batch_size)
    valid_gen = batch_generator(False, valid_wavs, y_valid, valid.word, silences, unknowns, batch_size=batch_size)

    start = time.time()
    model.fit_generator(
        generator=train_gen,
        epochs=30,
        steps_per_epoch=len_train // batch_size,
        validation_data=valid_gen,
        validation_steps=len_valid // batch_size,
        callbacks=[
            model_checkpoint,
            early_stopping,
            reduce_lr,
            # lr_tracker
        ],
        workers=4,
        use_multiprocessing=False,
        verbose=1)
    end = time.time()
    print('trained model in {}'.format(end-start))

    del train, valid, y_train, y_valid
    gc.collect()

    # model = load_model('model.model')
    # # validate(model, label_index, valid_data_path)
    # #
    # lp = LineProfiler()
    # lp_wrapper = lp(get_predicts)
    # index, results = lp_wrapper(model, label_index)
    # lp.print_stats()
    #
    # df = pd.DataFrame(columns=['fname', 'label'])
    # df['fname'] = index
    # df['label'] = results
    # df.to_csv(os.path.join(out_path, 'sub.csv'), index=False)

from keras import backend as K

class LearningRateTracker(Callback):
    def on_epoch_end(self, epoch, logs={}):
        optimizer = self.model.optimizer
        lr = K.eval(optimizer.lr * (1. / (1. + optimizer.decay * optimizer.iterations)))
        print('\nLR: {:.6f}\n'.format(lr))


if __name__ == "__main__":
    main()