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

BATCH_SIZE = 128
L = 16000
legal_labels = 'yes no up down left right on off stop go silence unknown'.split()
legal_labels_without_unknown = 'yes no up down left right on off stop go silence'.split()
root_path = r'..'
out_path = r'.'
model_path = r'.'
train_data_path = os.path.join(root_path, 'input', 'train', 'audio')
valid_data_path = os.path.join(root_path, 'input', 'train', 'valid')
test_internal_data_path = os.path.join(root_path, 'input', 'train', 'test')
test_data_path = os.path.join(root_path, 'input', 'test', 'audio')
background_noise_paths = glob(os.path.join(train_data_path, r'_background_noise_/*' + '.wav'))
silence_paths = glob(os.path.join(train_data_path, r'silence/*' + '.wav'))

def get_predicts(fpaths, model, label_index):
    label_index[~np.isin(label_index, legal_labels)] = 'unknown'
    index = []
    results = []
    batch = 128
    for fnames, imgs in tqdm(test_data_generator(fpaths, batch), total=math.ceil(len(fpaths) / batch)):
        predicted_probabilities = model.predict(imgs)
        print(np.sum(predicted_probabilities))
        predicts = []
        for i in range(len(predicted_probabilities)):
            if max(predicted_probabilities[i]) > 0.5:
                predicts.extend([label_index[np.argmax(predicted_probabilities[i])]])
            else:
                predicts.extend(['unknown'])
                print(fnames[i])
        index.extend(fnames)
        results.extend(predicts)
    return index, results

def get_data():
    train = prepare_data(get_path_label_df(train_data_path))
    valid = prepare_data(get_path_label_df(valid_data_path))
    silence_paths = train.path[train.word == 'silence']
    # train.drop(train[~train.word.isin(legal_labels_without_unknown)].index, inplace=True)
    # valid.drop(valid[~valid.word.isin(legal_labels_without_unknown)].index, inplace=True)
    train.reset_index(inplace=True)
    valid.reset_index(inplace=True)

    len_train = len(train.word.values)
    temp = label_transform(np.concatenate((train.word.values, valid.word.values)))
    y_train, y_valid = temp[:len_train], temp[len_train:]

    label_index = y_train.columns.values
    y_train = np.array(y_train.values)
    y_valid = np.array(y_valid.values)

    return train, valid, y_train, y_valid, label_index, silence_paths

def get_callbacks(model_name='model'):
    model_checkpoint = ModelCheckpoint(model_name + '.model', monitor='val_acc', save_best_only=True, save_weights_only=False,
                                       verbose=1)
    early_stopping = EarlyStopping(monitor='val_acc', patience=10, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=1, verbose=1)
    tensorboard = TensorBoard(log_dir='./' + model_name + 'logs', write_graph=True)
    lr_tracker = LearningRateTracker()
    return [model_checkpoint, early_stopping, reduce_lr, tensorboard, lr_tracker]

def train_model():
    train, valid, y_train, y_valid, label_index, silence_paths = get_data()

    rand_silence_paths = silence_paths.iloc[np.random.randint(0, len(silence_paths), 500)]
    silences = np.array(list(map(load_wav_by_path, rand_silence_paths)))

    # model = load_model('model.model')
    model = get_model(classes=31)
    # model.load_weights('model.model')
    train_gen = batch_generator_paths(train.path.values, y_train, train.word, silences, batch_size=BATCH_SIZE)
    valid_gen = batch_generator_paths(valid.path.values, y_valid, valid.word, silences, batch_size=BATCH_SIZE)
    model.fit_generator(
        generator=train_gen,
        epochs=100,
        steps_per_epoch=len(y_train) // BATCH_SIZE // 4,
        validation_data=valid_gen,
        validation_steps=len(y_valid) // BATCH_SIZE // 4,
        callbacks=get_callbacks('model2'),
        workers=4,
        use_multiprocessing=False,
        verbose=1
    )

def validate_predictions():
    _, _, _, _, label_index, _ = get_data()
    model = load_model('model2.model')
    valid = prepare_data(get_path_label_df(test_internal_data_path))
    y_true = np.array(valid.word.values)
    y_true[~np.isin(y_true, legal_labels)] = 'unknown'
    _, results = get_predicts(valid.path.values, model, label_index)
    labels = legal_labels
    # labels = next(os.walk(train_data_path))[1]
    confusion = confusion_matrix(y_true, results, labels)
    confusion_df = pd.DataFrame(confusion, index=labels, columns=labels)
    plt.figure(figsize=(10, 7))
    sn.heatmap(confusion_df, annot=True, fmt="d")
    plt.show()

def make_predictions():
    _, _, _, _, label_index, _ = get_data()
    model = load_model('model2.model')
    fpaths = glob(os.path.join(test_data_path, '*wav'))
    # fpaths = [os.path.join(test_data_path, 'clip_03cafe2bd.wav')]
    fpaths = np.random.choice(fpaths, 5000)
    index, results = get_predicts(fpaths, model, label_index)

    df = pd.DataFrame(columns=['fname', 'label'])
    df['fname'] = index
    df['label'] = results
    df.to_csv(os.path.join(out_path, 'sub.csv'), index=False)

def main():
    train_model()
    # validate_predictions()
    # make_predictions()

if __name__ == "__main__":
    main()