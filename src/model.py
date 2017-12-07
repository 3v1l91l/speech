from keras import optimizers, losses, activations, models, applications
from keras.layers import GRU, Reshape, GlobalAveragePooling2D, Convolution2D, Dense, Input, Flatten, Dropout, MaxPooling2D, BatchNormalization, ZeroPadding2D, Activation, Conv2D
from keras.models import Sequential, Model
from depthwise_conv2d import DepthwiseConvolution2D

def get_model():
    # input_shape = (99, 161, 1)
    # input_shape = (49, 161, 1)
    # input_shape = (101, 40, 1)
    # input_shape = (40, 101, 1)
    # input_shape = (128, 16, 1)
    # input_shape = (20, 16, 1)
    input_shape = (40, 51, 1)

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
    # img_1 = Dropout(rate=2)(img_1)
    # # img_1 = Convolution2D(32, kernel_size=3, activation=activations.relu)(img_1)
    # # img_1 = MaxPooling2D(pool_size=(2, 2))(img_1)
    # # img_1 = Dropout(rate=0.2)(img_1)
    # img_1 = Flatten()(img_1)
    #
    # dense_1 = BatchNormalization()(Dense(128, activation=activations.relu)(img_1))
    # dense_1 = BatchNormalization()(Dense(128, activation=activations.relu)(dense_1))
    # dense_1 = Dense(nclass, activation=activations.softmax)(dense_1)
    # model = models.Model(inputs=inp, outputs=dense_1)


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
    # model.add(Conv2D(32, kernel_size=(2, 2), activation='relu', input_shape=input_shape))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))
    # model.add(Flatten())
    # model.add(Dense(128, activation='relu'))
    # model.add(Dropout(0.25))
    # model.add(Dense(12, activation='softmax'))

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

    # model = Sequential()
    # model.add(ZeroPadding2D((1, 1), input_shape=input_shape))
    # model.add(
    #     Convolution2D(nb_filter=64, nb_row=3, nb_col=3, border_mode='valid', activation='relu', init='glorot_normal'))
    # model.add(ZeroPadding2D((1, 1)))
    # model.add(Convolution2D(64, 3, 3, activation='relu'))
    # model.add(MaxPooling2D((2, 2), strides=(1, 1)))
    # model.add(Dropout(0.25))
    #
    # model.add(ZeroPadding2D((1, 1)))
    # model.add(Convolution2D(nb_filter=64, nb_row=3, nb_col=3, border_mode='valid', activation='relu'))
    # model.add(ZeroPadding2D((1, 1)))
    # model.add(Convolution2D(64, 3, 3, activation='relu'))
    # model.add(MaxPooling2D((2, 2), strides=(1, 1)))
    # model.add(Dropout(0.25))
    #
    # model.add(ZeroPadding2D((1, 1)))
    # model.add(Convolution2D(nb_filter=32, nb_row=3, nb_col=3, border_mode='valid', activation='relu'))
    # model.add(ZeroPadding2D((1, 1)))
    # model.add(Convolution2D(32, 3, 3, activation='relu'))
    # model.add(MaxPooling2D((2, 2), strides=(1, 1)))
    # model.add(Dropout(0.25))
    #
    # model.add(Flatten())
    # model.add(Dense(2, init='glorot_normal'))
    # model.add(Activation('softmax'))

    # opt = optimizers.Adam()
    # opt = optimizers.SGD(lr  =0.02, decay=1e-6, momentum=0.9, nesterov=True)
    # opt = optimizers.SGD(lr=10*0.02)
    # base_model = applications.VGG16(weights=None, include_top=False, input_shape=(51,51,3))

    # # add a global spatial average pooling layer
    # x = base_model.output
    # x = GlobalAveragePooling2D()(x)
    # # let's add a fully-connected layer
    # x = Dense(1024, activation='relu')(x)
    # # and a logistic layer -- let's say we have 200 classes
    # predictions = Dense(12, activation='softmax')(x)
    #
    # # this is the model we will train
    # model = Model(inputs=base_model.input, outputs=predictions)

    opt = optimizers.Adam(lr=0.0001)
    # opt = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

    model.compile(optimizer=opt, loss=losses.categorical_crossentropy, metrics=['accuracy'])
    return model


def MobileNet(alpha=1, shallow=True, classes=12):
    input_shape = (40, 51, 1)
    # input_shape = (40, 51)
    img_input = Input(shape=input_shape)

    # x = Convolution2D(int(32 * alpha), (3, 3), strides=(2, 2), padding='same', use_bias=False)(img_input)
    # x = BatchNormalization()(x)
    # x = Activation('relu')(x)
    #
    # x = DepthwiseConvolution2D(int(32 * alpha), (3, 3), strides=(1, 1), padding='same', use_bias=False)(x)
    # x = BatchNormalization()(x)
    # x = Activation('relu')(x)
    # x = Convolution2D(int(64 * alpha), (1, 1), strides=(1, 1), padding='same', use_bias=False)(x)
    # x = BatchNormalization()(x)
    # x = Activation('relu')(x)
    # x = Dropout(0.25)(x)
    #
    # x = DepthwiseConvolution2D(int(64 * alpha), (3, 3), strides=(2, 2), padding='same', use_bias=False)(x)
    # x = BatchNormalization()(x)
    # x = Activation('relu')(x)
    # x = Convolution2D(int(64 * alpha), (1, 1), strides=(1, 1), padding='same', use_bias=False)(x)
    # x = BatchNormalization()(x)
    # x = Activation('relu')(x)
    # x = Dropout(0.25)(x)

    # x = GlobalAveragePooling2D()(x)


    # model = Sequential()
    # model.add(GRU(units=250, input_shape=input_shape))
    # model.add(Dense(classes, activation='softmax'))

    model = Sequential()
    model.add(Conv2D(filters=100,kernel_size=(4,10),strides=(1,2), activation='relu', input_shape=input_shape))
    conv_to_rnn_dims = (37,21*100)
    model.add(Reshape(target_shape=conv_to_rnn_dims, name='reshape'))
    model.add(GRU(units=136,activation='relu'))
    # model.add(Reshape(target_shape=conv_to_rnn_dims, name='reshape'))
    # model.add(GRU(units=136,activation='relu'))
    model.add(Dense(188, activation='relu'))
    model.add(Dense(classes, activation='softmax'))

    # opt = optimizers.Adam(lr=0.003)
    opt = optimizers.SGD(lr=0.3)

    model.compile(optimizer=opt, loss=losses.categorical_crossentropy, metrics=['accuracy'])
    return model
