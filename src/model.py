from keras import optimizers, losses, activations, models, applications
from keras.layers import TimeDistributed, Bidirectional, SeparableConv2D, GRU, Reshape, GlobalAveragePooling2D, Convolution2D, Dense, Input, Flatten, Dropout, MaxPooling2D, BatchNormalization, ZeroPadding2D, Activation, Conv2D
from keras.models import Sequential, Model
import keras.layers as layers
from keras.layers.advanced_activations import ELU
import keras
from keras import backend as K
from keras.models import Model
from keras.layers import (Input, Lambda, BatchNormalization)
from keras.regularizers import l2

def get_model_simple(classes=12):
    input_shape = (98, 40, 1)
    input = Input(shape=input_shape)
    num = 512
    x = Conv2D(num, (10, 4), strides=(2, 1), use_bias=False)(input)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)

    x = SeparableConv2D(num, kernel_size=3, strides=2, padding='same', use_bias=False)(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)

    x = SeparableConv2D(num, kernel_size=3, strides=1, padding='same', use_bias=False)(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)


    x = SeparableConv2D(num, kernel_size=3, strides=1, padding='same', use_bias=False)(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)

    x = SeparableConv2D(num, kernel_size=3, strides=1, padding='same', use_bias=False)(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)

    x = SeparableConv2D(num, kernel_size=3, strides=1, padding='same', use_bias=False)(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)


    x = GlobalAveragePooling2D()(x)

    if classes == 2:
        loss = losses.binary_crossentropy
        x = Dense(classes, activation='sigmoid')(x)
    else:
        loss = losses.categorical_crossentropy
        x = Dense(classes, activation='softmax')(x)

    model = Model(input, x)
    opt = optimizers.Adam(lr=0.005)
    model.compile(optimizer=opt, loss=losses.categorical_hinge, metrics=['categorical_accuracy'])
    return model

def get_model(classes=12):
    weight_decay = 1e-4
    input_shape = (99, 40, 1)
    input = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), strides=(2, 2), use_bias=False, name='block1_conv1')(input)
    x = BatchNormalization(name='block1_conv1_bn')(x)
    x = Activation('relu', name='block1_conv1_act')(x)
    x = Conv2D(64, (3, 3), use_bias=False, name='block1_conv2')(x)
    x = BatchNormalization(name='block1_conv2_bn')(x)
    x = Activation('relu', name='block1_conv2_act')(x)

    residual = Conv2D(128, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False, name='block2_sepconv1')(x)
    x = BatchNormalization(name='block2_sepconv1_bn')(x)
    x = Activation('relu', name='block2_sepconv2_act')(x)
    x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False, name='block2_sepconv2')(x)
    x = BatchNormalization(name='block2_sepconv2_bn')(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block2_pool')(x)
    x = layers.add([x, residual])

    residual = Conv2D(256, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = Activation('relu', name='block3_sepconv1_act')(x)
    x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False, name='block3_sepconv1')(x)
    x = BatchNormalization(name='block3_sepconv1_bn')(x)
    x = Activation('relu', name='block3_sepconv2_act')(x)
    x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False, name='block3_sepconv2')(x)
    x = BatchNormalization(name='block3_sepconv2_bn')(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block3_pool')(x)
    x = layers.add([x, residual])

    residual = Conv2D(728, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = Activation('relu', name='block4_sepconv1_act')(x)
    x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name='block4_sepconv1')(x)
    x = BatchNormalization(name='block4_sepconv1_bn')(x)
    x = Activation('relu', name='block4_sepconv2_act')(x)
    x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name='block4_sepconv2')(x)
    x = BatchNormalization(name='block4_sepconv2_bn')(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block4_pool')(x)
    x = layers.add([x, residual])

    for i in range(1):
        residual = x
        prefix = 'block' + str(i + 5)

        x = Activation('relu', name=prefix + '_sepconv1_act')(x)
        x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name=prefix + '_sepconv1')(x)
        x = BatchNormalization(name=prefix + '_sepconv1_bn')(x)
        x = Activation('relu', name=prefix + '_sepconv2_act')(x)
        x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name=prefix + '_sepconv2')(x)
        x = BatchNormalization(name=prefix + '_sepconv2_bn')(x)
        x = Activation('relu', name=prefix + '_sepconv3_act')(x)
        x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name=prefix + '_sepconv3')(x)
        x = BatchNormalization(name=prefix + '_sepconv3_bn')(x)

        x = layers.add([x, residual])

    residual = Conv2D(1024, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = Activation('relu', name='block13_sepconv1_act')(x)
    x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name='block13_sepconv1')(x)
    x = BatchNormalization(name='block13_sepconv1_bn')(x)
    x = Activation('relu', name='block13_sepconv2_act')(x)
    x = SeparableConv2D(1024, (3, 3), padding='same', use_bias=False, name='block13_sepconv2')(x)
    x = BatchNormalization(name='block13_sepconv2_bn')(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block13_pool')(x)
    x = layers.add([x, residual])

    x = SeparableConv2D(1536, (3, 3), padding='same', use_bias=False, name='block14_sepconv1')(x)
    x = BatchNormalization(name='block14_sepconv1_bn')(x)
    x = Activation('relu', name='block14_sepconv1_act')(x)

    x = SeparableConv2D(
        2048, (3, 3), padding='same',
        use_bias=False, name='block14_sepconv2',
        depthwise_regularizer=keras.regularizers.l2(weight_decay),
        pointwise_regularizer=keras.regularizers.l2(weight_decay)
    )(x)
    x = BatchNormalization(name='block14_sepconv2_bn')(x)
    x = Activation('relu', name='block14_sepconv2_act')(x)

    x = GlobalAveragePooling2D(name='avg_pool')(x)

    # x = model.output
    x = Dropout(0.5)(x)

    opt = optimizers.Adam(lr=0.005)
    # opt = optimizers.Adadelta()
    if classes == 2:
        loss = losses.binary_crossentropy
        x = Dense(classes, activation='sigmoid', kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    else:
        loss = losses.categorical_crossentropy
        x = Dense(classes, activation='softmax', kernel_regularizer=keras.regularizers.l2(1e-4))(x)

    model = Model(input, x)
    model.compile(optimizer=opt, loss=loss, metrics=['categorical_accuracy'])
    return model

def get_gru_model(classes=2):
    input_shape = (99, 40, 1)
    input = Input(shape=input_shape)
    # x = ZeroPadding2D(padding=(0, 37))(input)
    # x = BatchNormalization(axis=2, name='bn_0_freq')(x)

    # Conv block 1
    # x = Convolution2D(64, 3, 3, border_mode='same', name='conv1')(input)
    # x = BatchNormalization(axis=3, mode=0, name='bn1')(x)
    # x = Activation('relu')(x)
    # x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool1')(x)
    # x = Dropout(0.1, name='dropout1')(x)
    #
    # # Conv block 2
    # x = Convolution2D(128, 3, 3, border_mode='same', name='conv2')(x)
    # x = BatchNormalization(axis=3, mode=0, name='bn2')(x)
    # x = Activation('relu')(x)
    # x = MaxPooling2D(pool_size=(3, 3), strides=(2, 3), name='pool2')(x)
    # x = Dropout(0.1, name='dropout2')(x)
    #
    # # Conv block 3
    # x = Convolution2D(128, 3, 3, border_mode='same', name='conv3')(x)
    # x = BatchNormalization(axis=3, mode=0, name='bn3')(x)
    # x = Activation('relu')(x)
    # x = MaxPooling2D(pool_size=(4, 4), strides=(2, 4), name='pool3')(x)
    # x = Dropout(0.1, name='dropout3')(x)


    # x = GRU(256, activation='relu', return_sequences=False, input_shape=input_shape)(input)
    # # x = GRU(256, return_sequences=False)(x)
    # x = Dropout(0.5)(x)
    # model = Sequential()
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

    input = Input(shape=input_shape)
    x = Conv2D(256, (1, 4), strides=(1, 2), use_bias=False)(input)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)

    x = SeparableConv2D(256, kernel_size=3, strides=1, padding='same', use_bias=False)(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)

    x = Conv2D(256, (1, 4), strides=(1, 2), use_bias=False)(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)

    x = SeparableConv2D(256, kernel_size=3, strides=1, padding='same', use_bias=False)(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)

    # x = SeparableConv2D(256, kernel_size=3, strides=1, padding='same', use_bias=False)(x)
    # x = Activation('relu')(x)
    # x = BatchNormalization()(x)
    # x = Dropout(0.25)(x)
    #
    # x = SeparableConv2D(256, kernel_size=3, strides=1, padding='same', use_bias=False)(x)
    # x = Activation('relu')(x)
    # x = BatchNormalization()(x)
    # x = Dropout(0.25)(x)


    # x = GlobalAveragePooling2D()(x)
    #
    # x = SeparableConv2D(256, kernel_size=3, strides=1, padding='same', use_bias=False)(x)
    # x = Activation('relu')(x)
    # x = BatchNormalization()(x)
    # x = MaxPooling2D(pool_size=(2, 2), strides=(1, 2), name='pool3')(x)
    # x = Dropout(0.1, name='dropout3')(x)
    # x = Dropout(0.25)(x)


    # x = SeparableConv2D(256, kernel_size=3, strides=1, padding='same', use_bias=False)(input)
    # x = Activation('relu')(x)
    # x = BatchNormalization()(x)
    # x = Dropout(0.25)(x)
    #
    # x = SeparableConv2D(256, kernel_size=3, strides=1, padding='same', use_bias=False)(input)
    # x = Activation('relu')(x)
    # x = BatchNormalization()(x)
    # x = Dropout(0.25)(x)
    #
    # x = SeparableConv2D(256, kernel_size=3, strides=1, padding='same', use_bias=False)(input)
    # x = Activation('relu')(x)
    # x = BatchNormalization()(x)
    # x = Dropout(0.25)(x)
    #
    # x = SeparableConv2D(256, kernel_size=3, strides=1, padding='same', use_bias=False)(input)
    # x = Activation('relu')(x)
    # x = BatchNormalization()(x)
    # x = Dropout(0.25)(x)
    # #


    # x = Reshape((96, 512))(x)
    # x = GRU(32, input_shape=input_shape, return_sequences=True, activation='relu')(x)
    # # x = Dropout(0.25)(x)
    # x = GRU(64, input_shape=input_shape, return_sequences=True, activation='relu')(x)
    # # x = Dropout(0.25)(x)
    # x = GRU(128, input_shape=input_shape, return_sequences=True, activation='relu')(x)
    # # x = Dropout(0.25)(x)
    # x = GRU(256, input_shape=input_shape, return_sequences=False, activation='relu')(x)
    # x = Dropout(0.25)(x)
    x = GlobalAveragePooling2D()(x)

    if classes == 2:
        loss = losses.binary_crossentropy
        x = Dense(classes, activation='sigmoid')(x)
        # model.add(Dense(classes, activation='sigmoid'))
    else:
        loss = losses.categorical_crossentropy
        x = Dense(classes, activation='softmax')(x)
        # model.add(Dense(classes, activation='softmax'))

    model = Model(input, x)

    # loss = losses.binary_crossentropy
    # x = Dense(classes, activation='softmax')(x)

    # model = Model(input, x)
    opt = optimizers.Adam(lr=0.005)
    model.compile(optimizer=opt, loss=loss, metrics=['categorical_accuracy'])
    return model


def get_some_model(classes=2):
    input_shape = (98, 40)
    input = Input(shape=input_shape)
    # x = Conv2D(32, (20, 5), strides=(8, 2), use_bias=False)(input)
    # x = Activation('relu')(x)
    # x = BatchNormalization()(x)
    #
    # # x = GlobalAveragePooling2D()(x)
    # x = Reshape((180, 32))(x)

    x = GRU(256, return_sequences=True)(input)
    x = Dropout(0.25)(x)
    x = GRU(256, return_sequences=True)(x)
    x = Dropout(0.25)(x)

    if classes == 2:
        loss = losses.binary_crossentropy
        # x = TimeDistributed(Dense(classes, activation='sigmoid'))(x)
        x = Dense(classes, activation='sigmoid', W_regularizer=l2(0.01))(x)
    else:
        loss = losses.categorical_crossentropy
        # x = TimeDistributed(Dense(classes, activation='softmax'))(x)
        x = Dense(classes, activation='softmax', W_regularizer=l2(0.01))(x)

    model = Model(input, x)
    opt = optimizers.Adam(lr=0.005)
    model.compile(optimizer=opt, loss=loss, metrics=['categorical_accuracy'])
    return model

