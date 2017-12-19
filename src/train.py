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
legal_labels = 'yes no up down left right on off stop go silence unknown'.split()
recognized_labels = 'yes no up down left right on off stop go'.split()
root_path = r'..'
out_path = r'.'
model_path = r'.'
train_data_path = os.path.join(root_path, 'input', 'train', 'audio')
valid_data_path = os.path.join(root_path, 'input', 'train', 'valid')
test_internal_data_path = os.path.join(root_path, 'input', 'train', 'test')
test_data_path = os.path.join(root_path, 'input', 'test', 'audio')
background_noise_paths = glob(os.path.join(train_data_path, r'_background_noise_/*' + '.wav'))
silence_paths = glob(os.path.join(train_data_path, r'silence/*' + '.wav'))
BATCH_SIZE = 128


def get_predicts(model, silence_model, silence_label_index, label_index):
    fpaths = glob(os.path.join(test_data_path, '*wav'))
    # fpaths = np.random.choice(fpaths, 1000)
    index = []
    results = []
    batch = 128
    for fnames, imgs in tqdm(test_data_generator(fpaths, batch), total=math.ceil(len(fpaths) / batch)):
        silence_predicted_probabilities = silence_model.predict(imgs)
        predicted_probabilities = model.predict(imgs)
        predicts = []
        silence_label_index_ix = np.where(silence_label_index == 'silence')
        for i in range(len(silence_predicted_probabilities)):
            if(silence_predicted_probabilities[i][silence_label_index_ix] > 0.8):
                predicts.extend(['silence'])
            elif(np.max(predicted_probabilities[i]) > 0.8):
                predicts.extend(label_index[np.argmax(predicted_probabilities[i])])
            else:
                predicts.extend(['unknown'])
        index.extend(fnames)
        results.extend(predicts)
    return index, results

def validate(model, label_index, path):
    batch = 128
    all_fpaths = glob(os.path.join(path, '*/*wav'))
    all_folders = next(os.walk(train_data_path))[1]
    all_correct_count = 0
    for f in all_folders:
        correct_count = 0
        fpaths = [fp for fp in all_fpaths if fp.split(os.sep)[-2] == f]
        for labels, imgs in valid_data_generator(fpaths, batch):
            predicts = model.predict(imgs)
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
    silence_paths = train.path[train.word == 'silence']
    train.drop(train[~train.word.isin(recognized_labels)].index, inplace=True)
    valid.drop(valid[~valid.word.isin(recognized_labels)].index, inplace=True)
    train.reset_index(inplace=True)
    valid.reset_index(inplace=True)

    len_train = len(train.word.values)
    temp = label_transform(np.concatenate((train.word.values, valid.word.values)))
    y_train, y_valid = temp[:len_train], temp[len_train:]

    label_index = y_train.columns.values
    y_train = np.array(y_train.values)
    y_valid = np.array(y_valid.values)

    return train, valid, y_train, y_valid, label_index, silence_paths

def get_data_silence_not_silence():
    train, valid = get_train_valid_df()

    len_train = len(train.word.values)
    train.loc[train.word != 'silence', 'word'] = ['unknown']
    valid.loc[valid.word != 'silence', 'word'] = ['unknown']
    temp = label_transform(np.concatenate((train.word.values, valid.word.values)))
    y_train, y_valid = temp[:len_train], temp[len_train:]

    label_index = y_train.columns.values
    y_train = np.array(y_train.values)
    y_valid = np.array(y_valid.values)

    return train, valid, y_train, y_valid, label_index

def get_callbacks(model_name='model'):
    model_checkpoint = ModelCheckpoint(model_name + '.model', monitor='val_acc', save_best_only=True, save_weights_only=False,
                                       verbose=1)
    early_stopping = EarlyStopping(monitor='val_acc', patience=10, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=1, verbose=1)
    tensorboard = TensorBoard(log_dir='./' + model_name + 'logs', write_graph=True)
    lr_tracker = LearningRateTracker()
    return [model_checkpoint, early_stopping, reduce_lr, tensorboard, lr_tracker]



def train_silence_model():
    train, valid, y_train, y_valid, label_index = get_data_silence_not_silence()
    silence_model = get_silence_model()
    train_gen = batch_generator_silence_paths(train.path.values, y_train, train.word, batch_size=BATCH_SIZE)
    valid_gen = batch_generator_silence_paths(valid.path.values, y_valid, valid.word, batch_size=BATCH_SIZE)
    silence_model.fit_generator(
        generator=train_gen,
        epochs=3,
        steps_per_epoch=len(y_train) // BATCH_SIZE // 4,
        validation_data=valid_gen,
        validation_steps=len(y_valid) // BATCH_SIZE // 4,
        callbacks=get_callbacks('model'),
        workers=4,
        use_multiprocessing=False,
        verbose=1
    )

def train_model():
    train, valid, y_train, y_valid, label_index, silence_paths = get_data()

    pool = Pool()
    rand_silence_paths = silence_paths.iloc[np.random.randint(0, len(silence_paths), 500)]
    silences = np.array(list(pool.imap(load_wav_by_path, rand_silence_paths)))

    # model = load_model('model.model')
    model = get_model()
    # model.load_weights('model.model')
    train_gen = batch_generator_paths(train.path.values, y_train, train.word, silences, batch_size=BATCH_SIZE)
    valid_gen = batch_generator_paths(valid.path.values, y_valid, valid.word, silences, batch_size=BATCH_SIZE)
    model.fit_generator(
        generator=train_gen,
        epochs=100,
        steps_per_epoch=len(y_train) // BATCH_SIZE // 4,
        validation_data=valid_gen,
        validation_steps=len(y_valid) // BATCH_SIZE // 4,
        callbacks=get_callbacks('model'),
        workers=4,
        use_multiprocessing=False,
        verbose=1
    )

def make_predictions():
    _, _, _, _, label_index, _ = get_data()
    _, _, _, _, silence_label_index = get_data_silence_not_silence()
    model = load_model('model.model')
    silence_model = load_model('silence_model.model')
    index, results = get_predicts(model, silence_model, silence_label_index, label_index)

    df = pd.DataFrame(columns=['fname', 'label'])
    df['fname'] = index
    df['label'] = results
    df.to_csv(os.path.join(out_path, 'sub.csv'), index=False)

def validate_predictions():
    train, valid, y_train, y_valid, label_index = get_data()
    model = load_model('model.model')
    validate(model, label_index, test_internal_data_path)

def main():
    # train_silence_model()
    train_model()
    # validate_predictions()
    # make_predictions()

if __name__ == "__main__":
    main()