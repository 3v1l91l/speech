from keras import optimizers, losses, activations, models, applications
from keras.layers import SeparableConv2D, GRU, Reshape, GlobalAveragePooling2D, Convolution2D, Dense, Input, Flatten, Dropout, MaxPooling2D, BatchNormalization, ZeroPadding2D, Activation, Conv2D
from keras.models import Sequential, Model
from keras.layers.advanced_activations import ELU

def get_model(classes=12):
    input_shape = (98, 40, 1)
    input = Input(shape=input_shape)
    # x = Conv2D(256, (10, 4), strides=(2, 2), use_bias=False)(input)
    # x = Activation('relu')(x)
    # x = BatchNormalization()(x)
    # x = Dropout(0.25)(x)
    #
    # x = SeparableConv2D(256, kernel_size=3, strides=2, padding='same', use_bias=False)(x)
    # x = Activation('relu')(x)
    # x = BatchNormalization()(x)
    # x = Dropout(0.25)(x)
    #
    # x = SeparableConv2D(256, kernel_size=3, strides=1, padding='same', use_bias=False)(x)
    # x = Activation('relu')(x)
    # x = BatchNormalization()(x)
    # x = Dropout(0.25)(x)
    #
    # x = SeparableConv2D(256, kernel_size=3, strides=1, padding='same', use_bias=False)(x)
    # x = Activation('relu')(x)
    # x = BatchNormalization()(x)
    # x = Dropout(0.25)(x)
    #
    # x = SeparableConv2D(256, kernel_size=3, strides=1, padding='same', use_bias=False)(x)
    # x = Activation('relu')(x)
    # x = BatchNormalization()(x)
    # x = Dropout(0.25)(x)

    # x = Conv2D(32, (3, 3), strides=(2, 2), use_bias=False)(input)
    # x = BatchNormalization()(x)
    # x = Activation('relu')(x)
    # x = MaxPooling2D((3, 3), strides=(1, 1))(x)
    #
    # x = SeparableConv2D(128, (3, 3), use_bias=False)(x)
    # x = BatchNormalization()(x)
    # x = Activation('relu')(x)
    # x = SeparableConv2D(128, (3, 3), use_bias=False)(x)
    # x = BatchNormalization()(x)
    #
    # x = MaxPooling2D((3, 3), strides=(1, 1))(x)
    #
    # x = SeparableConv2D(256, (3, 3), use_bias=False)(x)
    # x = BatchNormalization()(x)
    # x = Activation('relu')(x)
    # x = SeparableConv2D(256, (3, 3), use_bias=False)(x)
    # x = BatchNormalization()(x)
    #
    # x = MaxPooling2D((3, 3), strides=(1, 1))(x)
    #
    # x = SeparableConv2D(512, (3, 3), use_bias=False)(x)
    # x = BatchNormalization()(x)
    # x = Activation('relu')(x)
    # x = SeparableConv2D(512, (3, 3), use_bias=False)(x)
    # x = BatchNormalization()(x)
    #
    # x = GlobalAveragePooling2D()(x)
    # x = Dense(classes, activation='softmax')(x)
    #
    # model = Model(input, x)

    nb_filters = 32  # number of convolutional filters to use
    pool_size = (2, 2)  # size of pooling area for max pooling
    kernel_size = (3, 3)  # convolution kernel size
    nb_layers = 4

    model = Sequential()
    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                            border_mode='valid', input_shape=input_shape))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))

    for layer in range(nb_layers - 1):
        model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
        model.add(BatchNormalization(axis=1))
        model.add(ELU(alpha=1.0))
        model.add(MaxPooling2D(pool_size=pool_size))
        model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(classes))
    model.add(Activation("softmax"))


    opt = optimizers.Adam(lr=0.005)
    loss = losses.categorical_crossentropy
    if classes == 2:
        loss = losses.binary_crossentropy
    model.compile(optimizer=opt, loss=loss, metrics=['accuracy'])
    return model


def get_silence_model(classes=2):
    input_shape = (98, 40, 1)
    input = Input(shape=input_shape)
    x = Conv2D(256, (10, 4), strides=(2, 2), use_bias=False)(input)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)

    x = SeparableConv2D(256, kernel_size=3, strides=2, padding='same', use_bias=False)(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)

    x = SeparableConv2D(256, kernel_size=3, strides=1, padding='same', use_bias=False)(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)

    x = GlobalAveragePooling2D()(x)
    x = Dense(classes, activation='sigmoid')(x)

    model = Model(input, x)
    opt = optimizers.Adam(lr=0.005)
    model.compile(optimizer=opt, loss=losses.binary_crossentropy, metrics=['accuracy'])
    return model
