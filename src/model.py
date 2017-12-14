from keras import optimizers, losses, activations, models, applications
from keras.layers import SeparableConv2D, GRU, Reshape, GlobalAveragePooling2D, Convolution2D, Dense, Input, Flatten, Dropout, MaxPooling2D, BatchNormalization, ZeroPadding2D, Activation, Conv2D
from keras.models import Sequential, Model

def get_model(classes=12):
    input_shape = (40, 101, 1)
    input = Input(shape=input_shape)
    x = Conv2D(256, (4, 10), strides=(2, 2), use_bias=False)(input)
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

    x = SeparableConv2D(256, kernel_size=3, strides=1, padding='same', use_bias=False)(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)

    x = SeparableConv2D(500, kernel_size=3, strides=1, padding='same', use_bias=False)(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)

    x = GlobalAveragePooling2D()(x)
    x = Dense(classes, activation='softmax')(x)

    model = Model(input, x)
    opt = optimizers.Adam(lr=0.005)

    model.compile(optimizer=opt, loss=losses.categorical_crossentropy, metrics=['accuracy'])
    return model

def get_silence_model(classes=2):
    input_shape = (40, 101, 1)
    input = Input(shape=input_shape)
    x = Conv2D(256, (4, 10), strides=(2, 2), use_bias=False)(input)
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
