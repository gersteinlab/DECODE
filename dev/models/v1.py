
import tensorflow as tf

import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Conv2D, MaxPooling2D, BatchNormalization, Flatten, GlobalAveragePooling2D, Multiply
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras.utils import Sequence, plot_model
from keras.constraints import unit_norm
from keras import regularizers


def SqueezeExcite(tensor, ratio=16):
    nb_channel = K.int_shape(tensor)[-1]

    x = GlobalAveragePooling2D()(tensor)
    x = Dense(nb_channel // ratio, activation='relu')(x)
    x = Dense(nb_channel, activation='sigmoid')(x)

    x = Multiply()([tensor, x])
    return x


def create_model(width=200):
    K.clear_session()
    pool2_list = []
    merge_list = []

    input_size = Input(shape=(5, width, 1))
    conv1_ = Conv2D(128, (3, 10), padding='same',activation='relu')(input_size)
    conv1  = SqueezeExcite(conv1_)
    conv2_ = Conv2D(64, (3, 1), padding='same',activation='relu')(conv1)
    conv2  = SqueezeExcite(conv2_)
    conv3_ = Conv2D(64, (3, 3), padding='same',activation='relu')(conv2)
    conv3  = SqueezeExcite(conv3_)
    conv4_ = Conv2D(128, (3, 1), padding='same',activation='relu')(conv3)
    conv4  = SqueezeExcite(conv4_)
    pool1  = MaxPooling2D(pool_size=(2, 2))(conv4)
    conv5_ = Conv2D(64, (3, 3), padding='same',activation='relu')(pool1)
    conv5  = SqueezeExcite(conv5_)
    conv6_ = Conv2D(64, (3, 3), padding='same',activation='relu')(conv5)
    conv6  = SqueezeExcite(conv6_)
    conv7_ = Conv2D(128, (3, 1), padding='same',activation='relu')(conv6)
    conv7  = SqueezeExcite(conv7_)
    pool2  = MaxPooling2D(pool_size=(2, 2))(conv7)

    x = Flatten()(pool2)
    dense1_ = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-5, l2=1e-4))
    dense1  = dense1_(x)
    x = Dropout(0.4)(dense1)
    dense2  = Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-5, l2=1e-4))(x)
    x = Dropout(0.4)(dense2)
    dense3 = Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-5, l2=1e-4))(x)
    x = Dropout(0.4)(dense3)
    pred_output = Dense(1, activation='sigmoid')(dense3)
    model = Model(input=[input_size], output=[pred_output])
    #model.summary()

    return model