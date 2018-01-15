from keras import optimizers, losses, activations, models, applications
from keras.layers import merge, TimeDistributed, Bidirectional, SeparableConv2D, GRU, Reshape, GlobalAveragePooling2D, Convolution2D, Dense, Input, Flatten, Dropout, MaxPooling2D, BatchNormalization, ZeroPadding2D, Activation, Conv2D
from keras.models import Sequential, Model
import keras.layers as layers
from keras.layers.advanced_activations import ELU
import keras
from keras import backend as K
from keras.models import Model
from keras.layers import (Input, Lambda, BatchNormalization)
from keras.regularizers import l2
import tpe
import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from keras.callbacks import Callback


class LearningRateTracker(Callback):
    def on_epoch_end(self, epoch, logs=None):
        lr = self.model.optimizer.lr
        decay = self.model.optimizer.decay
        iterations = self.model.optimizer.iterations
        lr_with_decay = lr / (1. + decay * K.cast(iterations, K.dtype(decay)))
        print("Learning rate: {}".format(K.eval(lr_with_decay)))

def get_callbacks(label_index, model_name='model'):
    model_checkpoint = ModelCheckpoint(model_name + '.model', monitor='val_custom_accuracy_in', save_best_only=True, save_weights_only=False,
                                       verbose=1)
    early_stopping = EarlyStopping(monitor='val_custom_accuracy_in', patience=5, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_custom_accuracy_in', factor=0.5, patience=0, verbose=1)
    tensorboard = TensorBoard(log_dir='./' + model_name + 'logs', write_graph=True)

    # model_checkpoint = ModelCheckpoint(model_name + '.model', monitor='val_acc', save_best_only=True,
    #                                    save_weights_only=False,
    #                                    verbose=1)
    # early_stopping = EarlyStopping(monitor='val_acc', patience=7, verbose=1)
    # reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=0, verbose=1)
    # tensorboard = TensorBoard(log_dir='./' + model_name + 'logs', write_graph=True)
    lr_tracker = LearningRateTracker()
    return [model_checkpoint, early_stopping, reduce_lr, tensorboard, lr_tracker]

def get_model_simple(label_index, classes=12):
    input_shape = (98, 40, 1)
    # input_shape = (101, 40, 1)
    input = Input(shape=input_shape)
    num = 256
    dropout = 0.3
    x = Conv2D(num, (10, 4), strides=(2, 1), use_bias=False)(input)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    # x = Dropout(dropout)(x)

    x = SeparableConv2D(num, kernel_size=3, strides=2, padding='same', use_bias=False)(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    # x = Dropout(dropout)(x)

    x = SeparableConv2D(num, kernel_size=3, strides=1, padding='same', use_bias=False)(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    # x = Dropout(dropout)(x)
    #
    x = SeparableConv2D(num, kernel_size=3, strides=1, padding='same', use_bias=False)(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    # x = Dropout(dropout)(x)

    x = SeparableConv2D(num, kernel_size=3, strides=1, padding='same', use_bias=False)(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    # x = Dropout(dropout)(x)

    x = SeparableConv2D(num, kernel_size=3, strides=1, padding='same', use_bias=False)(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    # x = Dropout(dropout)(x)

    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)

    # x = Dense(256)(x)
    # x = Activation('relu')(x)


    # if classes == 2:
    #     # loss = losses.binary_crossentropy
    #     x = Dense(classes, activation='sigmoid')(x)
    # else:
    #     # loss = losses.categorical_crossentropy
    #     # loss = custom_categorical_crossentropy
    #     x = Dense(classes, activation='softmax')(x)
    #     # x = Dense(classes, activation='sigmoid')(x)
    x = Dense(classes, activation='softmax')(x)
    # x = Dense(classes, activation='sigmoid')(x)

    model = Model(input, x)
    # opt = optimizers.Adam(lr=0.0012)
    opt = optimizers.Adam(lr=0.005)
    # model.compile(optimizer=opt, loss=custom_loss(label_index), metrics=[custom_accuracy(label_index)])
    # model.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
    # model.compile(optimizer=opt, loss=losses.binary_crossentropy(), metrics=[custom_accuracy(label_index)])
    model.compile(optimizer=opt, loss=losses.binary_crossentropy, metrics=['acc'])
    # model.compile(optimizer=opt, loss=losses.binary_crossentropy, metrics=['acc'])

    print(model.metrics_names)
    return model

def custom_loss(label_index):
    def custom_loss_in(y_true,y_pred):
        z = np.zeros(len(label_index), dtype=bool)
        z[label_index == ['unknown']] = True
        var = K.constant(np.array(z), dtype='float32')
        y_pred = K.switch(K.less(K.max(y_pred), K.variable(np.array(0.9), dtype='float32')), y_pred * var, y_pred)

        return K.mean(K.categorical_crossentropy(y_true, y_pred), axis=-1)
        # return K.categorical_crossentropy(y_true, y_pred)
        # return categorical_hinge(y_true, y_pred)

    return custom_loss_in
def squared_hinge(y_true, y_pred):
    return K.mean(K.square(K.maximum(1. - y_true * y_pred, 0.)), axis=-1)

def categorical_hinge(y_true, y_pred):
    pos = K.sum(y_true * y_pred, axis=-1)
    neg = K.max((1. - y_true) * y_pred, axis=-1)
    return K.maximum(0., neg - pos + 1.)

def custom_accuracy(label_index):
    def custom_accuracy_in(y_true, y_pred):
        z = np.zeros(len(label_index), dtype=bool)
        z[label_index == ['unknown']] = True
        var = K.constant(np.array(z), dtype='float32')
        y_pred2 = y_pred * var
        y_pred = K.switch(K.less(K.max(y_pred), K.variable(np.array(0.9), dtype='float32')), y_pred2, y_pred)
        y_pred = K.print_tensor(y_pred)
        return K.cast(K.equal(K.argmax(y_true, axis=-1), K.argmax(y_pred, axis=-1)), K.floatx())
    return custom_accuracy_in

# def custom_accuracy(label_index):
#     def custom_accuracy_in(y_true, y_pred):
#         # z = np.zeros(len(label_index), dtype=bool)
#         # z[label_index == ['unknown']] = True
#         # var = K.constant(np.array(z), dtype='float32')
#         # y_pred2 = y_pred * var
#         # y_pred = K.switch(K.less(K.max(y_pred), K.variable(np.array(0.8), dtype='float32')), y_pred2, y_pred)
#
#         return K.mean(K.equal(y_true, K.round(y_pred)), axis=-1)
#     return custom_accuracy_in

def get_model(label_index, classes=12):
    input_shape = (98, 40, 1)
    weight_decay = 1e-5
    dropout = 0.3
    input = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), strides=(2, 2), use_bias=False, name='block1_conv1', W_regularizer=keras.regularizers.l2(weight_decay))(input)
    x = BatchNormalization(name='block1_conv1_bn')(x)
    x = Activation('relu', name='block1_conv1_act')(x)
    # x = Dropout(dropout)(x)
    x = Conv2D(64, (3, 3), use_bias=False, name='block1_conv2', W_regularizer=keras.regularizers.l2(weight_decay))(x)
    x = BatchNormalization(name='block1_conv2_bn')(x)
    x = Activation('relu', name='block1_conv2_act')(x)
    # x = Dropout(dropout)(x)

    residual = Conv2D(128, (1, 1), strides=(2, 2), padding='same', use_bias=False, W_regularizer=keras.regularizers.l2(weight_decay))(x)
    residual = BatchNormalization()(residual)
    # x = Dropout(dropout)(x)

    x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False, name='block2_sepconv1', W_regularizer=keras.regularizers.l2(weight_decay))(x)
    x = BatchNormalization(name='block2_sepconv1_bn')(x)
    x = Activation('relu', name='block2_sepconv2_act')(x)
    # x = Dropout(dropout)(x)
    x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False, name='block2_sepconv2', W_regularizer=keras.regularizers.l2(weight_decay))(x)
    x = BatchNormalization(name='block2_sepconv2_bn')(x)
    # x = Dropout(dropout)(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block2_pool')(x)
    x = layers.add([x, residual])

    residual = Conv2D(256, (1, 1), strides=(2, 2), padding='same', use_bias=False, W_regularizer=keras.regularizers.l2(weight_decay))(x)
    residual = BatchNormalization()(residual)
    # x = Dropout(dropout)(x)

    x = Activation('relu', name='block3_sepconv1_act')(x)
    x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False, name='block3_sepconv1', W_regularizer=keras.regularizers.l2(weight_decay))(x)
    x = BatchNormalization(name='block3_sepconv1_bn')(x)
    x = Activation('relu', name='block3_sepconv2_act')(x)
    # x = Dropout(dropout)(x)
    x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False, name='block3_sepconv2', W_regularizer=keras.regularizers.l2(weight_decay))(x)
    x = BatchNormalization(name='block3_sepconv2_bn')(x)
    # x = Dropout(dropout)(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block3_pool')(x)
    x = layers.add([x, residual])
    # x = Dropout(dropout)(x)

    residual = Conv2D(728, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = Activation('relu', name='block4_sepconv1_act')(x)
    x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name='block4_sepconv1', W_regularizer=keras.regularizers.l2(weight_decay))(x)
    x = BatchNormalization(name='block4_sepconv1_bn')(x)
    # x = Dropout(dropout)(x)

    x = Activation('relu', name='block4_sepconv2_act')(x)
    x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name='block4_sepconv2', W_regularizer=keras.regularizers.l2(weight_decay))(x)
    x = BatchNormalization(name='block4_sepconv2_bn')(x)
    # x = Dropout(dropout)(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block4_pool')(x)
    x = layers.add([x, residual])

    for i in range(1):
        residual = x
        prefix = 'block' + str(i + 5)

        x = Activation('relu', name=prefix + '_sepconv1_act')(x)
        x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name=prefix + '_sepconv1', W_regularizer=keras.regularizers.l2(weight_decay))(x)
        x = BatchNormalization(name=prefix + '_sepconv1_bn')(x)
        x = Activation('relu', name=prefix + '_sepconv2_act')(x)
        x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name=prefix + '_sepconv2', W_regularizer=keras.regularizers.l2(weight_decay))(x)
        x = BatchNormalization(name=prefix + '_sepconv2_bn')(x)
        x = Activation('relu', name=prefix + '_sepconv3_act')(x)
        x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name=prefix + '_sepconv3', W_regularizer=keras.regularizers.l2(weight_decay))(x)
        x = BatchNormalization(name=prefix + '_sepconv3_bn')(x)

        x = layers.add([x, residual])
    #
    # residual = Conv2D(1024, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    # residual = BatchNormalization()(residual)
    #
    # x = Activation('relu', name='block13_sepconv1_act')(x)
    # x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name='block13_sepconv1')(x)
    # x = BatchNormalization(name='block13_sepconv1_bn')(x)
    # x = Activation('relu', name='block13_sepconv2_act')(x)
    # x = SeparableConv2D(1024, (3, 3), padding='same', use_bias=False, name='block13_sepconv2')(x)
    # x = BatchNormalization(name='block13_sepconv2_bn')(x)
    #
    # x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block13_pool')(x)
    # x = layers.add([x, residual])
    #
    # x = SeparableConv2D(1536, (3, 3), padding='same', use_bias=False, name='block14_sepconv1')(x)
    # x = BatchNormalization(name='block14_sepconv1_bn')(x)
    # x = Activation('relu', name='block14_sepconv1_act')(x)
    #
    # x = SeparableConv2D(
    #     2048, (3, 3), padding='same',
    #     use_bias=False, name='block14_sepconv2',
    #     # depthwise_regularizer=keras.regularizers.l2(weight_decay),
    #     # pointwise_regularizer=keras.regularizers.l2(weight_decay)
    # )(x)
    # x = BatchNormalization(name='block14_sepconv2_bn')(x)
    # x = Activation('relu', name='block14_sepconv2_act')(x)

    x = GlobalAveragePooling2D(name='avg_pool')(x)

    # x = model.output
    # x = Dropout(0.5)(x)

    # opt = optimizers.Adam(lr=0.005)
    # # opt = optimizers.Adadelta()
    # if classes == 2:
    #     loss = losses.binary_crossentropy
    #     x = Dense(classes, activation='sigmoid', kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    # else:
    #     loss = losses.categorical_crossentropy
    #     x = Dense(classes, activation='softmax', kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = Dropout(0.5)(x)

    # x = Dense(classes, activation='sigmoid')(x)
    x = Dense(classes, activation='softmax')(x)

    model = Model(input, x)
    opt = optimizers.Adam(lr=0.005)
    # opt = optimizers.Adam(lr=0.0005)
    # model.compile(optimizer=opt, loss=keras.losses.binary_crossentropy, metrics=[custom_accuracy(label_index)])

    model.compile(optimizer=opt, loss=custom_loss(label_index), metrics=[custom_accuracy(label_index)])

    # model = Model(input, x)
    # model.compile(optimizer=opt, loss=loss, metrics=['categorical_accuracy'])
    return model

