from keras import optimizers, losses, activations, models
from keras.layers import Convolution2D, Dense, Input, Flatten, Dropout, MaxPooling2D, BatchNormalization

def get_model():
    # input_shape = (99, 161, 1)
    input_shape = (49, 161, 1)
    # input_shape = (101, 40, 1)
    # input_shape = (40, 101, 1)

    nclass = 12
    inp = Input(shape=input_shape)
    norm_inp = BatchNormalization()(inp)
    img_1 = Convolution2D(8, kernel_size=2, activation=activations.relu)(norm_inp)

    img_1 = Convolution2D(8, kernel_size=2, activation=activations.relu)(img_1)
    img_1 = MaxPooling2D(pool_size=(2, 2))(img_1)
    img_1 = Dropout(rate=0.2)(img_1)
    img_1 = Convolution2D(16, kernel_size=3, activation=activations.relu)(img_1)
    img_1 = Convolution2D(16, kernel_size=3, activation=activations.relu)(img_1)
    img_1 = MaxPooling2D(pool_size=(2, 2))(img_1)
    img_1 = Dropout(rate=0.2)(img_1)
    img_1 = Convolution2D(32, kernel_size=3, activation=activations.relu)(img_1)
    img_1 = MaxPooling2D(pool_size=(2, 2))(img_1)
    img_1 = Dropout(rate=0.2)(img_1)
    img_1 = Flatten()(img_1)

    dense_1 = BatchNormalization()(Dense(128, activation=activations.relu)(img_1))
    dense_1 = BatchNormalization()(Dense(128, activation=activations.relu)(dense_1))
    dense_1 = Dense(nclass, activation=activations.softmax)(dense_1)

    # nclass = 12
    # inp = Input(shape=input_shape)
    # norm_inp = BatchNormalization()(inp)
    # img_1 = Convolution2D(8*2, kernel_size=2, activation=activations.relu)(norm_inp)
    #
    # img_1 = Convolution2D(8*2, kernel_size=2, activation=activations.relu)(img_1)
    # img_1 = MaxPooling2D(pool_size=(2, 2))(img_1)
    # img_1 = Dropout(rate=0.2)(img_1)
    # img_1 = Convolution2D(16*2, kernel_size=3, activation=activations.relu)(img_1)
    # img_1 = Convolution2D(16*2, kernel_size=3, activation=activations.relu)(img_1)
    # img_1 = MaxPooling2D(pool_size=(2, 2))(img_1)
    # img_1 = Dropout(rate=0.2)(img_1)
    # img_1 = Convolution2D(32*2, kernel_size=3, activation=activations.relu)(img_1)
    # img_1 = MaxPooling2D(pool_size=(2, 2))(img_1)
    # img_1 = Dropout(rate=0.2)(img_1)
    # img_1 = Flatten()(img_1)
    #
    # dense_1 = BatchNormalization()(Dense(128, activation=activations.relu)(img_1))
    # dense_1 = BatchNormalization()(Dense(128, activation=activations.relu)(dense_1))
    # dense_1 = Dense(nclass, activation=activations.softmax)(dense_1)


    model = models.Model(inputs=inp, outputs=dense_1)
    # opt = optimizers.Adam()
    sgd = optimizers.SGD(lr=3e-2, decay=1e-6, momentum=0.9, nesterov=True)

    model.compile(optimizer=sgd, loss=losses.categorical_crossentropy, metrics=['accuracy'])
    return model