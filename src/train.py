import os
import numpy as np
from scipy import signal
from glob import glob
import re
import pandas as pd
import gc
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from keras.models import load_model
from tqdm import tqdm
from multiprocessing import Pool
import time
import random
from lib import *
import scipy.io.wavfile as wavfile
import cProfile
import math
from generator import *
# import lightgbm as lgb

from model import *

L = 16000
batch_size = 128
silence_labels = ['silence']
legal_labels = 'yes no up down left right on off stop go silence unknown'.split()
known_labels = 'yes no up down left right on off stop go'.split()
root_path = r'..'
out_path = r'.'
model_path = r'.'
train_data_path = os.path.join(root_path, 'input', 'train', 'audio')
valid_data_path = os.path.join(root_path, 'input', 'train', 'valid')
test_internal_data_path = os.path.join(root_path, 'input', 'train', 'test')
test_data_path = os.path.join(root_path, 'input', 'test', 'audio')
background_noise_paths = glob(os.path.join(train_data_path, r'_background_noise_/*' + '.wav'))
silence_paths = glob(os.path.join(train_data_path, r'silence/*' + '.wav'))

def get_predicts(model, label_index, cool_guys):
    fpaths = glob(os.path.join(test_data_path, '*wav'))
    # fpaths = np.random.choice(fpaths, 1000)
    index = []
    results = []
    batch = 128
    for fnames, imgs in tqdm(test_data_generator(fpaths, batch), total=math.ceil(len(fpaths) / batch)):
        predicted_probabilities = model.predict(imgs)
        predict_max_indexes = []
        silence_label_index = int((label_index == 'silence').nonzero()[0])
        for predicted_probability in predicted_probabilities:
            predict_max_index = np.argmax(predicted_probability)
            # if predict_max_index == unknown_label_index:    # try to come up with real label to replicate label distribution
            #     predicted_probability_without_unknown = predicted_probability
            #     predicted_probability_without_unknown[unknown_label_index] = 0
            #     if(any(predicted_probability_without_unknown > 0.3)):
            #         print(max(predicted_probability))
            #         predict_max_index = np.argmax(predicted_probability_without_unknown)
            if max(predicted_probability) < 0.2:
                cool_guys += 1
                predict_max_index = silence_label_index
            predict_max_indexes.append(predict_max_index)
        predicts = [label_index[p] for p in predict_max_indexes]

        index.extend(fnames)
        results.extend(predicts)
    print('coll guys: %s' % cool_guys)
    return index, results

def validate(model, label_index, path):
    zz = label_index == 'unknown'
    batch = 128
    all_fpaths = glob(os.path.join(path, '*/*wav'))
    all_folders = next(os.walk(train_data_path))[1]
    all_correct_count = 0
    for f in all_folders:
        correct_count = 0
        fpaths = [fp for fp in all_fpaths if fp.split(os.sep)[-2] == f]
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

def get_train_valid_df():
    train = prepare_data(get_path_label_df(train_data_path))
    valid = prepare_data(get_path_label_df(valid_data_path))
    return train, valid

def get_data():
    train, valid = get_train_valid_df()

    len_train = len(train.word.values)
    temp = label_transform(np.concatenate((train.word.values, valid.word.values)))
    y_train, y_valid = temp[:len_train], temp[len_train:]

    label_index = y_train.columns.values
    y_train = np.array(y_train.values)
    y_valid = np.array(y_valid.values)

    return train, valid, y_train, y_valid, label_index

def get_data_silence_not_silence():
    train, valid = get_train_valid_df()

    len_train = len(train.word.values)
    train.word[train.word != 'silence'] = 'not_silence'
    valid.word[valid.word != 'silence'] = 'not_silence'
    temp = label_transform(np.concatenate((train.word.values, valid.word.values)))
    y_train, y_valid = temp[:len_train], temp[len_train:]

    label_index = y_train.columns.values
    y_train = np.array(y_train.values)
    y_valid = np.array(y_valid.values)

    return train, valid, y_train, y_valid, label_index

def train_model():
    train, valid, y_train, y_valid, label_index = get_data()
    y_train_words = train.word
    y_valid_words = valid.word
    len_train = len(train.word.values)
    len_valid = len(valid.word.values)
    early_stopping = EarlyStopping(monitor='val_acc', patience=10, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=1, verbose=1)
    lr_tracker = LearningRateTracker()
    # tensorboard = TensorBoard(log_dir='./logs', write_graph=True)

    ## read wavs beforehand
    pool = Pool()
    # train_wavs = np.array(list(pool.imap(load_wav_by_path, train.path.values)))
    # valid_wavs = np.array(list(pool.imap(load_wav_by_path, valid.path.values)))
    silence_paths = train.path[train.word == 'silence']
    rand_silence_paths = silence_paths.iloc[np.random.randint(0, len(silence_paths), 500)]
    silences = np.array(list(pool.imap(load_wav_by_path, rand_silence_paths)))

    unknown_paths = train.path[train.word == 'unknown']
    rand_unknown_paths = unknown_paths.iloc[np.random.randint(0, len(unknown_paths), 200)]
    unknowns = np.array(list(pool.imap(load_wav_by_path, rand_unknown_paths)))

    ## silence model
    train_silence, valid_silence, y_train_silence, y_valid_silence, label_index_silence = get_data_silence_not_silence()

    # y_train_words_silence = [y if y == 'silence' else 'not_silence' for y in y_train_words_silence]
    model_checkpoint = ModelCheckpoint('model_silence.model', monitor='val_acc', save_best_only=True, save_weights_only=False,
                                       verbose=1)
    model_silence = get_silence_model()
    train_gen = batch_generator_paths_silence(False, train_silence.path.values, y_train_silence, train_silence.word, silences, unknowns,
                                      batch_size=batch_size)
    valid_gen = batch_generator_paths_silence(True, valid_silence.path.values, y_valid_silence, valid_silence.word, silences, unknowns,
                                      batch_size=batch_size)
    model_silence.fit_generator(
        generator=train_gen,
        epochs=100,
        steps_per_epoch=len_train // batch_size // 4,
        validation_data=valid_gen,
        validation_steps=len_valid // batch_size // 4,
        callbacks=[
            model_checkpoint,
            early_stopping,
            reduce_lr,
            lr_tracker,
            # tensorboard
        ],
        workers=4,
        use_multiprocessing=False,
        verbose=1)

    # model = load_model('model.model')
    #
    model = get_model()
    # model.load_weights('model.model')
    model_checkpoint = ModelCheckpoint('model.model', monitor='val_acc', save_best_only=True, save_weights_only=False,
                                       verbose=1)







    # train_gen = batch_generator(True, train_wavs, y_train, train.word, silences, unknowns, batch_size=batch_size)
    # valid_gen = batch_generator(False, valid_wavs, y_valid, valid.word, silences, unknowns, batch_size=batch_size)
    train_gen = batch_generator_paths(False, train.path.values, y_train, train.word, silences, unknowns, batch_size=batch_size)
    valid_gen = batch_generator_paths(True, valid.path.values, y_valid, valid.word, silences, unknowns, batch_size=batch_size)

    model.fit_generator(
        generator=train_gen,
        epochs=100,
        steps_per_epoch=len_train // batch_size // 4,
        validation_data=valid_gen,
        validation_steps=len_valid // batch_size // 4,
        callbacks=[
            model_checkpoint,
            early_stopping,
            reduce_lr,
            lr_tracker,
            # tensorboard
        ],
        workers=4,
        use_multiprocessing=False,
        verbose=1)

def make_predictions():
    train, valid, y_train, y_valid, label_index = get_data()
    cool_guys = 0
    model = load_model('model.model')
    index, results = get_predicts(model, label_index, cool_guys)

    df = pd.DataFrame(columns=['fname', 'label'])
    df['fname'] = index
    df['label'] = results
    df.to_csv(os.path.join(out_path, 'sub.csv'), index=False)

def validate_predictions():
    train, valid, y_train, y_valid, label_index = get_data()
    model = load_model('model.model')
    validate(model, label_index, test_internal_data_path)

def main():
    train_model()
    # validate_predictions()
    # make_predictions()



if __name__ == "__main__":
    main()