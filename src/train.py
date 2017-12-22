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
from sklearn.metrics import confusion_matrix
from model import *
import seaborn as sn

L = 16000
legal_labels = 'yes no up down left right on off stop go silence unknown'.split()
legal_labels_without_silence = 'yes no up down left right on off stop go unknown'.split()
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

def get_predicts(fpaths, model, silence_model, label_index, silence_label_index):
    # fpaths = np.random.choice(fpaths, 1000)
    label_index[~np.isin(label_index, legal_labels)] = 'unknown'
    index = []
    results = []
    batch = 128
    for fnames, imgs in tqdm(test_data_generator(fpaths, batch), total=math.ceil(len(fpaths) / batch)):
        silence_predicted_probabilities = silence_model.predict(imgs)
        predicted_probabilities = model.predict(imgs)
        predicts = []
        silence_label_index_ix = np.where(silence_label_index == 'silence')
        unknown_label_index_ix = np.where(silence_label_index == 'unknown')
        for i in range(len(fnames)):
            if(np.argmax(silence_predicted_probabilities[i]) == silence_label_index_ix):
                predicts.extend(['silence'])
            elif(np.argmax(predicted_probabilities[i]) > 0.5):
                predicts.extend([label_index[np.argmax(predicted_probabilities[i])]])
            else:
                predicts.extend(['unknown'])

        index.extend(fnames)
        results.extend(predicts)
    return index, results

def validate(path, model, silence_model, label_index, silence_label_index):
    valid = prepare_data(get_path_label_df(path))
    y_true = np.array(valid.word.values)
    y_true[~np.isin(y_true, legal_labels)] = 'unknown'
    _, y_pred = get_predicts(valid.path.values, model, label_index)
    labels = legal_labels
    # labels = next(os.walk(train_data_path))[1]
    confusion = confusion_matrix(y_true, y_pred, labels)
    confusion_df = pd.DataFrame(confusion, index=labels, columns=labels)
    plt.figure(figsize=(10, 7))
    sn.heatmap(confusion_df, annot=True, fmt="d")
    plt.show()

def get_train_valid_df():
    train = prepare_data(get_path_label_df(train_data_path))
    valid = prepare_data(get_path_label_df(valid_data_path))
    return train, valid

def get_data():
    train, valid = get_train_valid_df()
    silence_paths = train.path[train.word == 'silence']
    train.drop(train[train.word.isin(['silence'])].index, inplace=True)
    valid.drop(valid[valid.word.isin(['silence'])].index, inplace=True)
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
    silence_paths = train.path[train.word == 'silence']
    unknown_paths = train.path[train.word != 'silence']


    len_train = len(train.word.values)
    train.loc[train.word != 'silence', 'word'] = ['unknown']
    valid.loc[valid.word != 'silence', 'word'] = ['unknown']
    temp = label_transform(np.concatenate((train.word.values, valid.word.values)))
    y_train, y_valid = temp[:len_train], temp[len_train:]

    label_index = y_train.columns.values
    y_train = np.array(y_train.values)
    y_valid = np.array(y_valid.values)

    return train, valid, y_train, y_valid, label_index, silence_paths, unknown_paths

def get_callbacks(model_name='model'):
    model_checkpoint = ModelCheckpoint(model_name + '.model', monitor='val_acc', save_best_only=True, save_weights_only=False,
                                       verbose=1)
    early_stopping = EarlyStopping(monitor='val_acc', patience=10, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=1, verbose=1)
    tensorboard = TensorBoard(log_dir='./' + model_name + 'logs', write_graph=True)
    lr_tracker = LearningRateTracker()
    return [model_checkpoint, early_stopping, reduce_lr, tensorboard, lr_tracker]


def train_silence_model():
    train, valid, y_train, y_valid, label_index, silence_paths, unknown_paths = get_data_silence_not_silence()
    rand_silence_paths = silence_paths.iloc[np.random.randint(0, len(silence_paths), 500)]
    silences = np.array(list(map(load_wav_by_path, rand_silence_paths)))
    rand_unknown_paths = silence_paths.iloc[np.random.randint(0, len(silence_paths), 500)]
    unknowns = np.array(list(map(load_wav_by_path, rand_unknown_paths)))

    silence_model = get_model(classes=2)
    train_gen = batch_generator_silence_paths(train.path.values, y_train, train.word, silences, unknowns, batch_size=BATCH_SIZE)
    valid_gen = batch_generator_silence_paths(valid.path.values, y_valid, valid.word, silences, unknowns, batch_size=BATCH_SIZE)
    silence_model.fit_generator(
        generator=train_gen,
        epochs=10,
        steps_per_epoch=len(y_train) // BATCH_SIZE // 4,
        validation_data=valid_gen,
        validation_steps=len(y_valid) // BATCH_SIZE // 4,
        callbacks=get_callbacks('model_silence'),
        workers=4,
        use_multiprocessing=False,
        verbose=1
    )

def train_model():
    train, valid, y_train, y_valid, label_index, silence_paths = get_data()

    rand_silence_paths = silence_paths.iloc[np.random.randint(0, len(silence_paths), 500)]
    silences = np.array(list(map(load_wav_by_path, rand_silence_paths)))

    # model = load_model('model.model')
    model = get_model(classes=30)
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
    _, _, _, _, silence_label_index, _ = get_data_silence_not_silence()
    model = load_model('model.model')
    silence_model = load_model('model_silence.model')
    fpaths = glob(os.path.join(test_data_path, '*wav'))
    index, results = get_predicts(fpaths, model, silence_model, silence_label_index, label_index)

    df = pd.DataFrame(columns=['fname', 'label'])
    df['fname'] = index
    df['label'] = results
    df.to_csv(os.path.join(out_path, 'sub.csv'), index=False)

def make_predictions_old():
    _, _, _, _, label_index, _, _ = get_data_old()
    model = load_model('model_old.model')
    fpaths = glob(os.path.join(test_data_path, '*wav'))
    fpaths = np.random.choice(fpaths, 5000)
    index, results = get_predicts_old(fpaths, model, label_index)

    df = pd.DataFrame(columns=['fname', 'label'])
    df['fname'] = index
    df['label'] = results
    df.to_csv(os.path.join(out_path, 'sub.csv'), index=False)

def validate_predictions():
    _, _, _, _, label_index, _ = get_data()
    _, _, _, _, silence_label_index, _ = get_data_silence_not_silence()
    model = load_model('model.model')
    silence_model = load_model('model_silence.model')
    validate(test_internal_data_path, model, silence_model, label_index, silence_label_index)

def main():
    train_silence_model()
    # train_model()
    # validate_predictions()

if __name__ == "__main__":
    main()

