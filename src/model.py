from keras import optimizers, losses, activations, models
from keras.layers import Convolution2D, Dense, Input, Flatten, Dropout, MaxPooling2D, BatchNormalization, Activation
from keras.models import Sequential

def get_model():
    # input_shape = (99, 161, 1)
    input_shape = (49, 161, 1)
    # input_shape = (101, 40, 1)
    # input_shape = (40, 101, 1)

    # nclass = 12
    # inp = Input(shape=input_shape)
    # norm_inp = BatchNormalization()(inp)
    # img_1 = Convolution2D(8, kernel_size=2, activation=activations.relu)(norm_inp)
    #
    # img_1 = Convolution2D(8, kernel_size=2, activation=activations.relu)(img_1)
    # img_1 = MaxPooling2D(pool_size=(2, 2))(img_1)
    # img_1 = Dropout(rate=0.2)(img_1)
    # img_1 = Convolution2D(16, kernel_size=3, activation=activations.relu)(img_1)
    # img_1 = Convolution2D(16, kernel_size=3, activation=activations.relu)(img_1)
    # img_1 = MaxPooling2D(pool_size=(2, 2))(img_1)
    # img_1 = Dropout(rate=0.2)(img_1)
    # img_1 = Convolution2D(32, kernel_size=3, activation=activations.relu)(img_1)
    # img_1 = MaxPooling2D(pool_size=(2, 2))(img_1)
    # img_1 = Dropout(rate=0.2)(img_1)
    # img_1 = Flatten()(img_1)
    #
    # dense_1 = BatchNormalization()(Dense(128, activation=activations.relu)(img_1))
    # dense_1 = BatchNormalization()(Dense(128, activation=activations.relu)(dense_1))
    # dense_1 = Dense(nclass, activation=activations.softmax)(dense_1)

    nclass = 12
    inp = Input(shape=input_shape)
    norm_inp = BatchNormalization()(inp)
    img_1 = Convolution2D(8*2, kernel_size=2, activation=activations.relu)(norm_inp)

    img_1 = Convolution2D(8*2, kernel_size=2, activation=activations.relu)(img_1)
    img_1 = MaxPooling2D(pool_size=(2, 2))(img_1)
    img_1 = Dropout(rate=0.2)(img_1)
    img_1 = Convolution2D(16*2, kernel_size=3, activation=activations.relu)(img_1)
    img_1 = Convolution2D(16*2, kernel_size=3, activation=activations.relu)(img_1)
    img_1 = MaxPooling2D(pool_size=(2, 2))(img_1)
    img_1 = Dropout(rate=0.2)(img_1)
    img_1 = Convolution2D(32*2, kernel_size=3, activation=activations.relu)(img_1)
    img_1 = MaxPooling2D(pool_size=(2, 2))(img_1)
    img_1 = Dropout(rate=0.2)(img_1)
    img_1 = Flatten()(img_1)

    dense_1 = BatchNormalization()(Dense(128, activation=activations.relu)(img_1))
    dense_1 = BatchNormalization()(Dense(128, activation=activations.relu)(dense_1))
    dense_1 = Dense(nclass, activation=activations.softmax)(dense_1)


    model = models.Model(inputs=inp, outputs=dense_1)


    # model = Sequential()
    # model.add(Convolution2D(128, 5, border_mode='valid', input_shape=input_shape))
    # model.add(MaxPooling2D(pool_size=(1, 4), strides=(1, 4)))
    # model.add(Flatten())
    # model.add(Dropout(0.4))
    # model.add(Dense(256))
    # model.add(Activation('relu'))
    # model.add(Dropout(0.2))
    # model.add(Dense(256))
    # model.add(Activation('relu'))
    # model.add(Dropout(0.3))
    # model.add(Dense(12))
    # model.add(Activation('softmax'))

    # opt = optimizers.Adam()
    # sgd = optimizers.SGD(lr  =5e-1, decay=1e-6, momentum=0.9, nesterov=True)
    sgd = optimizers.SGD(lr=0.02)
    model.compile(optimizer=sgd, loss=losses.categorical_crossentropy, metrics=['accuracy'])
    return model