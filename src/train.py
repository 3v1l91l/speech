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
from keras.models import Model
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

def get_predicts(fpaths, model, label_index):
    # fpaths = np.random.choice(fpaths, 1000)
    # label_index[~np.isin(label_index, legal_labels)] = 'unknown'
    index = []
    results = []
    batch = 128
    for fnames, imgs in tqdm(test_data_generator(fpaths, batch), total=math.ceil(len(fpaths) / batch)):
        predicted_probabilities = model.predict(imgs)
        # print(len(predicted_probabilities))
        predicts = []
        for i in range(len(fnames)):
            print(np.max(predicted_probabilities[i]))
            if(np.max(predicted_probabilities[i]) > 0.95):
                # print(np.max(predicted_probabilities[i]))
                predicts.extend([label_index[np.argmax(predicted_probabilities[i])]])
            else:
                predicts.extend(['unknown'])
            # predicts.extend([label_index[np.argmax(predicted_probabilities[i])]])

        index.extend(fnames)
        results.extend(predicts)
    return index, results

def validate(binary_label, path, model, label_index):
    valid = prepare_data(get_path_label_df(path))
    valid.loc[valid.word != binary_label, 'word'] = 'unknown'
    y_true = np.array(valid.word.values)
    ix = np.random.choice(range(len(y_true)), 15000)

    _, y_pred = get_predicts(valid.path.values[ix], model, label_index)
    # labels = next(os.walk(train_data_path))[1]
    confusion = confusion_matrix(y_true[ix], y_pred, label_index)
    confusion_df = pd.DataFrame(confusion, index=label_index, columns=label_index)
    plt.figure(figsize=(10, 7))
    sn.heatmap(confusion_df, annot=True, fmt="d")
    plt.show()

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
    _, _, _, _, label_index, _, _ = get_data()
    # model = get_model_simple(label_index, classes=12)
    # model.load_weights('model3.model')
    model = load_model('model3.model', custom_objects={'custom_accuracy_in': custom_accuracy(label_index), 'custom_loss_in': custom_loss(label_index)})

    fpaths = glob(os.path.join(test_data_path, '*wav'))
    fpaths = np.random.choice(fpaths, 1000)
    # fpaths = glob(os.path.join(test_data_path, 'clip_2e4ba4c25.wav'))
    index, results = get_predicts(fpaths, model, label_index)

    df = pd.DataFrame(columns=['fname', 'label'])
    df['fname'] = index
    df['label'] = results
    df.to_csv(os.path.join(out_path, 'sub.csv'), index=False)

def validate_predictions(binary_label):
    _, _, _, _, label_index, _, _ = get_data(binary_label)
    # model = load_model('model2.model')
    # model = get_model_simple(label_index, classes=12)
    model = load_model(binary_label+'.model')#, custom_objects={'custom_accuracy_in': custom_accuracy(label_index), 'custom_loss_in': custom_loss(label_index)})
    # model = get_model_simple(label_index, classes=12)

    validate(binary_label, test_internal_data_path, model, label_index)
#
# def get_emb(bottleneck, fpaths):
#     results = []
#     batch = 128
#     for fnames, imgs in tqdm(test_data_generator(fpaths, batch), total=math.ceil(len(fpaths) / batch)):
#         predicts = bottleneck.predict(imgs)
#         results.extend(predicts)
#     return results
#
# def train_tpe():
#     n_in = 256
#     n_out = 256
#
#     model = get_model_simple(31)
#     model.load_weights('model.model')
#     bottleneck = Bottleneck(model, ~1)
#     train, valid, y_train, y_valid, label_index, silence_paths, unknown_paths = get_data()
#     rand_silence_paths = silence_paths.iloc[np.random.randint(0, len(silence_paths), 500)]
#     silences = np.array(list(map(load_wav_by_path, rand_silence_paths)))
#
#     # train_emb = bottleneck.predict(train.path.values, batch_size=256)
#     # np.save('train_emb', train_emb)
#     train_emb = np.load('train_emb.npy')
#     # dev_emb = bottleneck.predict(valid.path.values, batch_size=256)
#     # np.save('dev_emb', dev_emb)
#     dev_emb = np.load('dev_emb.npy')
#
#
#     pca = PCA(n_out)
#     pca.fit(train_emb)
#     W_pca = pca.components_
#     tpe, tpe_pred = build_tpe(n_in, n_out, W_pca.T)
#     # tpe.load_weights('mineer.h5')
#
#     NB_EPOCH = 500000
#     COLD_START = NB_EPOCH
#     BATCH_SIZE = 4
#     BIG_BATCH_SIZE = 1000
#
#     z = np.zeros((BIG_BATCH_SIZE,))
#
#     dev_protocol = np.zeros((len(valid.word.values), len(valid.word.values)), dtype=np.bool)
#     for word in list(set(valid.word.values)):
#         word_true = valid.word.values == word
#         dev_protocol[np.outer(word_true, word_true)] = True
#
#     test(tpe_pred, dev_emb, dev_protocol)
#     # mineer = float('inf')
#     # for e in range(NB_EPOCH):
#     #     print('epoch: {}'.format(e))
#     #     a, p, n = get_triplet_batch(tpe_pred, train_emb, y_train, train.word, batch_size=BIG_BATCH_SIZE)
#     #     tpe.fit([a, p, n], z, batch_size=BATCH_SIZE, epochs=1)
#     #     if e !=0 and e%50 == 0:
#     #         eer = test(tpe_pred, dev_emb, dev_protocol)
#     #         print('EER: {:.2f}'.format(eer * 100))
#     #         if eer < mineer:
#     #             mineer = eer
#     #             tpe.save_weights('mineer.h5')
#
# def test(tpe_pred, dev_emb, dev_protocol):
#     dev_emb2 = tpe_pred.predict(dev_emb)
#     tsc, isc = get_scores(dev_emb2, dev_protocol)
#     eer, _, _, _ = calc_metrics(tsc, isc)
#     return eer
#
#
# def get_scores(data_y, protocol):
#     data_y = data_y / np.linalg.norm(data_y, axis=1)[:, np.newaxis]
#     scores = data_y @ data_y.T
#
#     return scores[protocol], scores[np.logical_not(protocol)]
#
#
# def calc_metrics(targets_scores, imposter_scores):
#     min_score = np.minimum(np.min(targets_scores), np.min(imposter_scores))
#     max_score = np.maximum(np.max(targets_scores), np.max(imposter_scores))
#
#     n_tars = len(targets_scores)
#     n_imps = len(imposter_scores)
#
#     N = 100
#
#     fars = np.zeros((N,))
#     frrs = np.zeros((N,))
#     dists = np.zeros((N,))
#
#     min_gap = float('inf')
#     eer = 0
#
#     for i, dist in enumerate(np.linspace(min_score, max_score, N)):
#         far = len(np.where(imposter_scores > dist)[0]) / n_imps
#         frr = len(np.where(targets_scores < dist)[0]) / n_tars
#         fars[i] = far
#         frrs[i] = frr
#         dists[i] = dist
#
#         gap = np.abs(far - frr)
#
#         if gap < min_gap:
#             min_gap = gap
#             eer = (far + frr) / 2
#
#     return eer, fars, frrs, dists


def main():
    # train_silence_model()
    for label in legal_labels_without_unknown:
        train_model(label)
    # train_model('no')
    # train_tpe()
    # train_model_unknown()
    # validate_predictions('down')
    # make_predictions()

if __name__ == "__main__":
    main()

