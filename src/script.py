from scipy.io import wavfile
import numpy as np
import pandas as pd
from scipy import signal
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from keras.models import Sequential, Model
from keras.layers import Input, Dense, TimeDistributed, LSTM, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten, Conv2D, GlobalMaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint
import matplotlib.pyplot as plt
from lib import get_path_label_df, prepare_data, get_specgrams
from keras import optimizers
from keras.models import load_model
# from predictor import predict_all
from keras.applications.vgg16 import VGG16
from keras.applications.inception_v3 import InceptionV3

train_words = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go', 'silence']
labels = train_words + ['unknown']
shape = (128, 201, 1)

def batch_generator(X, y, batch_size=16):
    '''
    Return batcho of random spectrograms and corresponding labels
    '''

    while True:
        # choose batch_size random images / labels from the data
        idx = np.random.randint(0, X.shape[0], batch_size)
        im = X[idx]
        label = y[idx]

        specgram = get_specgrams(im)
        yield np.concatenate([specgram]), label


def get_model(shape):
    '''
    Create a keras model.
    '''
    inputlayer = Input(shape=shape)
    #
    # model = BatchNormalization()(inputlayer)
    # model = Conv2D(16, ((2, 2)), activation='elu')(model)
    # model = Dropout(0.25)(model)
    # model = MaxPooling2D((2, 2))(model)
    #
    # model = Flatten()(model)
    # model = Dense(32, activation='elu')(model)
    # # model = Dropout(0.25)(model)
    #
    # model = Dense(len(labels), activation='softmax')(model)
    # # model = VGG16(include_top=False, weights=None, input_tensor=None, input_shape=None,
    # #                                pooling=None, classes=len(labels))
    #
    # model = Model(inputs=inputlayer, outputs=model)


    # nb_filters = 32  # number of convolutional filters to use
    # pool_size = (2, 2)  # size of pooling area for max pooling
    # kernel_size = ((2, 2))  # convolution kernel size
    # nb_layers = 4
    #
    # model = Sequential()
    # model.add(Conv2D(nb_filters, kernel_size,
    #                         padding='valid', input_shape=shape))
    # model.add(BatchNormalization(axis=1))
    # model.add(Activation('relu'))
    #
    # for layer in range(nb_layers - 1):
    #     model.add(Conv2D(nb_filters, kernel_size))
    #     model.add(BatchNormalization(axis=1))
    #     model.add(ELU(alpha=1.0))
    #     model.add(MaxPooling2D(pool_size=pool_size))
    #     model.add(Dropout(0.25))
    #
    # model.add(Flatten())
    # model.add(Dense(128))
    # model.add(Activation('relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(12))
    # model.add(Activation("sigmoid"))

    model = Sequential()
    model.add(Conv2D(64, (2, 2), padding='valid', input_shape=shape))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (2, 2)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (2, 2)))
    model.add(Activation('relu'))
    model.add(Conv2D(128, (2, 2)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(256, (2, 2)))
    model.add(Activation('relu'))
    model.add(Conv2D(256, (2, 2)))
    model.add(Activation('relu'))
    model.add(Conv2D(256, (2, 2)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(512, (2, 2)))
    model.add(Activation('relu'))
    model.add(Conv2D(512, (2, 2)))
    model.add(Activation('relu'))
    model.add(Conv2D(512, (2, 2)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(2000))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2000))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(12, activation='softmax'))

    # x = GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.

    # Create model.
    # model = Model(inputlayer, x, name='vgg16')

    return model

train = prepare_data(get_path_label_df('../input/train/audio/'))
model = get_model(shape)
sgd = optimizers.SGD(lr=0.012, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# create training and test data.

labelbinarizer = LabelBinarizer()

labelbinarizer.fit(labels)
y = labelbinarizer.transform(train.word)
X = train.path
X, Xt, y, yt = train_test_split(X, y, test_size=0.2, stratify=y)

batch_size = 128
train_gen = batch_generator(X.values, y, batch_size=batch_size)
valid_gen = batch_generator(Xt.values, yt, batch_size=batch_size)
model_checkpoint = ModelCheckpoint('model.model', monitor='val_acc', save_best_only=True, save_weights_only=False, verbose=1)
# model = load_model('model.model')

model.fit_generator(
    generator=train_gen,
    epochs=1,
    steps_per_epoch=X.shape[0] // batch_size,
    validation_data=valid_gen,
    validation_steps=Xt.shape[0] // batch_size,
    callbacks=[
        model_checkpoint
    ])

model = load_model('model.model')

# test = prepare_data(get_path_label_df('../input/test/'))
# paths = test.path.tolist()
# predictions = []
# # files = ['clip_1064e319a.wav', 'clip_26e1ae31b.wav']
# # paths = ['../input/test/audio/' + x for x in files]
#
# for path in paths[:100]:
#     specgram = get_specgrams([path], shape)
#     pred = model.predict(np.array(specgram))
#     predictions.extend(pred)
#
# labels = [labelbinarizer.inverse_transform(p.reshape(1, -1) == max(p), threshold=0.5)[0] for p in predictions]
# print(labels)
# test['labels'] = labels
# test.path = test.path.apply(lambda x: str(x).split('/')[-1])
# submission = pd.DataFrame({'fname': test.path.tolist(), 'label': labels})
# submission.to_csv('submission.csv', index=False)
