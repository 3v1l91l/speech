import os
import numpy as np
from scipy import signal
from glob import glob
import re
import pandas as pd
import gc
from keras.models import load_model
from tqdm import tqdm
from multiprocessing import Pool
import time
import random
from lib import *
import scipy.io.wavfile as wavfile
# import cProfile
import math
from generator import *
from sklearn.metrics import confusion_matrix
from model import *
import seaborn as sn
import matplotlib.pyplot as plt
from keras.models import Model, clone_model
from sklearn.model_selection import StratifiedKFold
from bottleneck import Bottleneck
# from identification import get_scores, calc_metrics

from sklearn.decomposition import PCA
L = 16000
legal_labels = 'yes no up down left right on off stop go silence unknown'.split()
legal_labels_without_silence = 'yes no up down left right on off stop go unknown'.split()
legal_labels_without_unknown = 'yes no up down left right on off stop go silence'.split()
recognized_labels = 'yes no up down left right on off stop go'.split()
legal_labels_without_unknown_and_silence = 'yes no up down left right on off stop go'.split()
legal_labels_without_unknown_can_be_flipped = [x for x in legal_labels_without_unknown_and_silence if x[::-1] not in legal_labels_without_unknown_and_silence]
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
import multiprocessing


def get_predicts(fpaths, models):
    # fpaths = np.random.choice(fpaths, 1000)
    # label_index[~np.isin(label_index, legal_labels)] = 'unknown'
    index = []
    results = []
    batch = 128
    label_index = np.array(list(models.keys()))
    for fnames, imgs in tqdm(test_data_generator(fpaths, batch), total=math.ceil(len(fpaths) / batch)):
        predictions = [model.predict(imgs) for (label,model) in models.items()]
        predictions = np.array([[p[0] for p in p]for p in predictions]).swapaxes(0, 1)
        prediction_labels = np.array(['unknown']*len(fnames))
        high_prob_lx = np.max(predictions, axis=1) > 0.8
        prediction_labels[high_prob_lx] = label_index[np.argmax(predictions[high_prob_lx], axis=1)]
        index.extend(fnames)
        results.extend(list(prediction_labels))
    return index, results

def validate(path, models):
    valid = prepare_data(get_path_label_df(path))
    # valid.loc[valid.word != binary_label, 'word'] = 'unknown'
    y_true = np.array(valid.word.values)
    # ix = np.random.choice(range(len(y_true)), 3000)

    _, y_pred = get_predicts(valid.path.values, models)
    # labels = next(os.walk(train_data_path))[1]
    # keys = list(models.keys())
    # keys.extend(['unknown'])
    confusion = confusion_matrix(y_true, y_pred, legal_labels)
    confusion_df = pd.DataFrame(confusion, index=legal_labels, columns=legal_labels)
    plt.figure(figsize=(10, 7))
    svm = sn.heatmap(confusion_df, annot=True, fmt="d")
    figure = svm.get_figure()
    figure.savefig('svm_conf.png', dpi=400)

def get_train_valid_df():
    train = prepare_data(get_path_label_df(train_data_path))
    valid = prepare_data(get_path_label_df(valid_data_path))
    return train, valid

def get_data(binary_label):
    train, valid = get_train_valid_df()
    silence_paths = train.path[train.word == 'silence']
    unknown_paths = train.path[train.word == 'unknown']
    if binary_label != 'silence':
        train.drop(train[train.word.isin(['silence'])].index, inplace=True)
        valid.drop(valid[valid.word.isin(['silence'])].index, inplace=True)
        train.reset_index(inplace=True)
        valid.reset_index(inplace=True)

    original_labels_train = np.array(train.word.values)
    original_labels_valid = np.array(valid.word.values)
    train.loc[train.word != binary_label, 'word'] = 'unknown'
    valid.loc[valid.word != binary_label, 'word'] = 'unknown'

    # len_train = len(train.word.values)
    # temp = label_transform(np.concatenate((train.word.values, valid.word.values)))
    # y_train, y_valid = temp[:len_train], temp[len_train:]

    y_train = np.zeros((len(train),2),dtype=np.uint8)
    y_train[train.word == binary_label,0] = 1
    y_train[train.word != binary_label, 1] = 1

    y_valid = np.zeros((len(valid),2),dtype=np.uint8)
    y_valid[valid.word == binary_label,0] = 1
    y_valid[valid.word != binary_label, 1] = 1

    label_index = np.array([binary_label, 'unknown'])
    # y_train = np.array(y_train.values)
    # y_valid = np.array(y_valid.values)

    return train, valid, y_train, y_valid, label_index, silence_paths, unknown_paths, original_labels_train, original_labels_valid

def train_model(binary_label):
    train, valid, y_train, y_valid, label_index, silence_paths, unknown_paths, original_labels_train, original_labels_valid = get_data(binary_label)

    rand_silence_paths = silence_paths.iloc[np.random.randint(0, len(silence_paths), 500)]
    silences = np.array(list(map(load_wav_by_path, rand_silence_paths)))
    rand_unknown_paths = unknown_paths.iloc[np.random.randint(0, len(unknown_paths), 500)]
    unknowns = np.array(list(map(load_wav_by_path, rand_unknown_paths)))

    # model = load_model('model3.model')
    # model = load_model('model3.model', custom_objects={'custom_accuracy_in': custom_accuracy(label_index), 'custom_loss_in': custom_loss(label_index)})

    # model = get_some_model(classes=12)
    model = get_model_simple(label_index, classes=2)
    # model = get_model_simple(label_index, classes=2)

    model.load_weights(binary_label+ '.model')

    # model = get_model(classes=30)
    # model = get_model_simple(classes=30)
    # model = get_some_model(classes=30)
    model.summary()
    # model.load_weights('model2.model')
    # train_gen = batch_generator_paths(train.path.values, y_train, train.word, silences, batch_size=BATCH_SIZE)
    # valid_gen = batch_generator_paths(valid.path.values, y_valid, valid.word, silences, batch_size=BATCH_SIZE)
    unknown_y = label_index == ['unknown']
    train_possible_unknown_ix = train.word[train.word != binary_label].index
    valid_possible_unknown_ix = valid.word[valid.word != binary_label].index
    train_possible_can_be_flipped_ix = train.word[np.isin(original_labels_train, legal_labels_without_unknown_can_be_flipped)].index
    valid_possible_can_be_flipped_ix = valid.word[np.isin(original_labels_valid, legal_labels_without_unknown_can_be_flipped)].index
    train_possible_known_ix = train.word[train.word == binary_label].index
    valid_possible_known_ix = valid.word[valid.word == binary_label].index
    train_gen = batch_generator_binary(False, binary_label, train.path.values, y_train, train.word, silences, unknowns,
                                       unknown_y, original_labels_train, train_possible_unknown_ix, train_possible_can_be_flipped_ix, train_possible_known_ix, batch_size=BATCH_SIZE)
    valid_gen = batch_generator_binary(True, binary_label, valid.path.values, y_valid, valid.word, silences, unknowns,
                                       unknown_y, original_labels_valid, valid_possible_unknown_ix, valid_possible_can_be_flipped_ix, valid_possible_known_ix, batch_size=BATCH_SIZE)
    model.fit_generator(
        generator=train_gen,
        epochs=100,
        steps_per_epoch=len(y_train) // BATCH_SIZE // 4,
        validation_data=valid_gen,
        validation_steps=len(y_valid) // BATCH_SIZE // 4,
        callbacks=get_callbacks(label_index, binary_label),
        workers=24,
        use_multiprocessing=False,
        verbose=1
    )

def make_predictions():
    models = dict()
    # model_empty = get_model_simple([], classes=2)
    for label in legal_labels_without_unknown:
        # print(label)
        # model = clone_model(model_empty)
        # model.load_weights(label + '.model')
        label_index = np.array([label, 'unknown'])
        model = load_model(label + '.model', custom_objects={'custom_accuracy_in': custom_accuracy(label_index), 'custom_loss_in': custom_loss(label_index)})
        models[label] = model
        
    fpaths = glob(os.path.join(test_data_path, '*wav'))

    # silences
    # fpaths = [os.path.join(test_data_path, 'clip_ff3eccdb8.wav'),
    #           os.path.join(test_data_path, 'clip_37f62e83c.wav'),
    #           os.path.join(test_data_path, 'clip_7adb2a420.wav'),
    #           os.path.join(test_data_path, 'clip_8b0fd6b46.wav'), # correct
    #           os.path.join(test_data_path, 'clip_0d17d07d0.wav'),
    #           os.path.join(test_data_path, 'clip_986b229a7.wav')  # eight
    #         ]
    # fpaths = np.random.choice(fpaths, 2000)
    index, results = get_predicts(fpaths[:80000], models)

    df = pd.DataFrame(columns=['fname', 'label'])
    df['fname'] = index
    df['label'] = results
    df.to_csv(os.path.join(out_path, 'sub.csv'), index=False)

def validate_predictions():
    models = dict()
    # model_empty = get_model_simple([], classes=2)
    for label in legal_labels_without_unknown:
        print(label)
        # model = clone_model(model_empty)
        # model.load_weights(label + '.model')
        label_index = np.array([label, 'unknown'])
        model = load_model(label + '.model', custom_objects={'custom_accuracy_in': custom_accuracy(label_index), 'custom_loss_in': custom_loss(label_index)})
        models[label] = model
    # models = {label: load_model(label + '.model') for label in legal_labels_without_unknown}
    validate(test_internal_data_path, models)

def main():
    # train_silence_model()
    # for label in 'off stop go'.split():
    # # for label in 'no right on'.split():
    #     train_model(label)
    # train_model('up')
    # train_tpe()
    # train_model_unknown()
    # validate_predictions()
    make_predictions()

if __name__ == "__main__":
    main()

