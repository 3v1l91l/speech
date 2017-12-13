from lib import *
import numpy as np
from multiprocessing import Pool

L = 16000
legal_labels = 'yes no up down left right on off stop go silence unknown'.split()
root_path = r'..'
out_path = r'.'
model_path = r'.'
train_data_path = os.path.join(root_path, 'input', 'train', 'audio')
valid_data_path = os.path.join(root_path, 'input', 'train', 'valid')
test_internal_data_path = os.path.join(root_path, 'input', 'train', 'test')
test_data_path = os.path.join(root_path, 'input', 'test', 'audio')
background_noise_paths = glob(os.path.join(train_data_path, r'_background_noise_/*' + '.wav'))
silence_paths = glob(os.path.join(train_data_path, r'silence/*' + '.wav'))

def get_data():
    train = prepare_data(get_path_label_df(train_data_path))
    valid = prepare_data(get_path_label_df(valid_data_path))

    len_train = len(train.word.values)
    temp = label_transform(np.concatenate((train.word.values, valid.word.values)))
    y_train, y_valid = temp[:len_train], temp[len_train:]

    label_index = y_train.columns.values
    y_train = np.array(y_train.values)
    y_valid = np.array(y_valid.values)


    return train, valid, y_train, y_valid, label_index

train, valid, y_train, y_valid, label_index = get_data()
y_train_word = np.array(train.word)
y_valid_word = np.array(valid.word)
pool = Pool()
train_wavs = np.array(list(pool.imap(load_wav_by_path, train.path.values)))
valid_wavs = np.array(list(pool.imap(load_wav_by_path, valid.path.values)))
