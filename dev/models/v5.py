
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
    x = Flatten()(input_size)
    dense1 = Dense(1024, activation='relu')(x)
    x = Dropout(0.4)(dense1)
    dense2 = Dense(32, activation='relu')(x)
    x = Dropout(0.4)(dense2)
    pred_output = Dense(1, activation='sigmoid')(x)
    model = Model(input=[input_size], output=[pred_output])
    model.summary()

    return model