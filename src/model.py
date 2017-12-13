from keras import optimizers, losses, activations, models, applications
from keras.layers import SeparableConv2D, GRU, Reshape, GlobalAveragePooling2D, Convolution2D, Dense, Input, Flatten, Dropout, MaxPooling2D, BatchNormalization, ZeroPadding2D, Activation, Conv2D
from keras.models import Sequential, Model

def get_model(classes=12):
    # input_shape = (10, 51, 1)

    # input = Input(shape=input_shape)
    # x = Conv2D(276, (4, 10), strides=(2, 2), use_bias=False)(input)
    # x = BatchNormalization()(x)
    # x = Activation('relu')(x)
    # # x = Dropout(0.25)(x)
    #
    # x = SeparableConv2D(276, kernel_size=3, strides=2, padding='same', use_bias=False)(x)
    # x = BatchNormalization()(x)
    # x = Activation('relu')(x)
    # # x = Dropout(0.25)(x)
    #
    # x = SeparableConv2D(276, kernel_size=3, strides=1, padding='same', use_bias=False)(x)
    # x = BatchNormalization()(x)
    # x = Activation('relu')(x)
    # # x = Dropout(0.25)(x)
    #
    # x = SeparableConv2D(276, kernel_size=3, strides=1, padding='same', use_bias=False)(x)
    # x = BatchNormalization()(x)
    # x = Activation('relu')(x)
    # # x = Dropout(0.25)(x)
    #
    # x = SeparableConv2D(276, kernel_size=3, strides=1, padding='same', use_bias=False)(x)
    # x = BatchNormalization()(x)
    # x = Activation('relu')(x)
    # # x = Dropout(0.25)(x)
    #
    # x = SeparableConv2D(276, kernel_size=3, strides=1, padding='same', use_bias=False)(x)
    # x = BatchNormalization()(x)
    # x = Activation('relu')(x)
    # # x = Dropout(0.25)(x)
    #
    # x = GlobalAveragePooling2D()(x)
    # x = Dense(classes, activation='softmax')(x)


    # input_shape = (40, 101, 1)
    #
    # input = Input(shape=input_shape)
    # x = Conv2D(64, (8, 20), strides=(1, 1), use_bias=True)(input)
    # x = Activation('relu')(x)
    # x = BatchNormalization()(x)
    # x = Dropout(0.5)(x)
    # x = MaxPooling2D(pool_size=(2, 2), strides=None, padding='same')(x)
    #
    # x = Conv2D(64, (4, 10), strides=(1, 1), use_bias=True)(x)
    # x = Activation('relu')(x)
    # x = BatchNormalization()(x)
    # x = Dropout(0.5)(x)
    # x = MaxPooling2D(pool_size=(2, 2), strides=None, padding='same')(x)
    #
    # x = Flatten()(x)
    # x = Dense(classes, activation='softmax')(x)
    # model = Model(input, x)

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

    x = SeparableConv2D(256, kernel_size=3, strides=1, padding='same', use_bias=False)(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)

    x = GlobalAveragePooling2D()(x)
    x = Dense(classes, activation='softmax')(x)

    model = Model(input, x)
    opt = optimizers.Adam(lr=0.005)
    # opt = optimizers.SGD(lr=0.1)

    model.compile(optimizer=opt, loss=losses.categorical_crossentropy, metrics=['accuracy'])
    return model
