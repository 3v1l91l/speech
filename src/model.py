from keras import optimizers, losses, activations, models, applications
from keras.layers import SeparableConv2D, GRU, Reshape, GlobalAveragePooling2D, Convolution2D, Dense, Input, Flatten, Dropout, MaxPooling2D, BatchNormalization, ZeroPadding2D, Activation, Conv2D
from keras.models import Sequential, Model

def get_model(classes=12):
    input_shape = (10, 51, 1)

    input = Input(shape=input_shape)
    x = Conv2D(276, (4, 10), strides=(2, 2), use_bias=False)(input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # x = Dropout(0.25)(x)

    x = SeparableConv2D(276, kernel_size=3, strides=2, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # x = Dropout(0.25)(x)

    x = SeparableConv2D(276, kernel_size=3, strides=1, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # x = Dropout(0.25)(x)

    x = SeparableConv2D(276, kernel_size=3, strides=1, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # x = Dropout(0.25)(x)

    x = SeparableConv2D(276, kernel_size=3, strides=1, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # x = Dropout(0.25)(x)

    x = SeparableConv2D(276, kernel_size=3, strides=1, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # x = Dropout(0.25)(x)

    x = GlobalAveragePooling2D()(x)
    x = Dense(classes, activation='softmax')(x)

    model = Model(input, x)

    opt = optimizers.Adam()
    model.compile(optimizer=opt, loss=losses.categorical_crossentropy, metrics=['accuracy'])
    return model
