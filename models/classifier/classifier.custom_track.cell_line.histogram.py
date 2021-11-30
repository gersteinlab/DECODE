import numpy as np
import string
import random
import os
from sklearn.model_selection import train_test_split
import pickle
import argparse
import math
from datetime import datetime
import sklearn

import tensorflow as tf
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, SpatialDropout2D, BatchNormalization, LSTM, concatenate, Activation, GlobalAveragePooling2D, Multiply
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras.utils import Sequence, plot_model
from keras.constraints import unit_norm
from keras import regularizers
from keras.callbacks import EarlyStopping, Callback, TensorBoard
from matplotlib import pyplot as plt
import keras_metrics as km


def auroc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred, curve="ROC", summation_method='careful_interpolation')[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc

def auprc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred, curve='PR', summation_method='careful_interpolation')[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def SqueezeExcite(tensor, ratio=16):
    nb_channel = K.int_shape(tensor)[-1]

    x = GlobalAveragePooling2D()(tensor)
    x = Dense(nb_channel // ratio, activation='relu')(x)
    x = Dense(nb_channel, activation='sigmoid')(x)

    x = Multiply()([tensor, x])
    return x

def load_model():
    K.clear_session()
    pool2_list = []
    merge_list = []
    input_list = []
    for track in tracks_list:
        input_size = Input(shape=(1, 100, 1))
        input_list.append(input_size)
        conv1_ = Activation(activation='relu')(BatchNormalization()(Conv2D(128, (1, 10), padding='same')(input_size)))
        conv1  = SqueezeExcite(conv1_)
        merge_list.append(conv1)

        conv2_ = Activation(activation='relu')(BatchNormalization()(Conv2D(64, (1, 1), padding='same')(conv1)))
        conv2  = SqueezeExcite(conv2_)
        conv3_ = Activation(activation='relu')(BatchNormalization()(Conv2D(64, (1, 3), padding='same')(conv2)))
        conv3  = SqueezeExcite(conv3_)
        conv4_ = Activation(activation='relu')(BatchNormalization()(Conv2D(128, (1, 1), padding='same')(conv3)))
        conv4  = SqueezeExcite(conv4_)
        pool1  = MaxPooling2D(pool_size=(1, 2))(conv4)
        conv5_ = Activation(activation='relu')(BatchNormalization()(Conv2D(64, (1, 3), padding='same')(pool1)))
        conv5  = SqueezeExcite(conv5_)
        conv6_ = Activation(activation='relu')(BatchNormalization()(Conv2D(64, (1, 3), padding='same')(conv5)))
        conv6  = SqueezeExcite(conv6_)
        conv7_ = Activation(activation='relu')(BatchNormalization()(Conv2D(128, (1, 1), padding='same')(conv3)))
        conv7  = SqueezeExcite(conv7_)

        pool2  = MaxPooling2D(pool_size=(1, 2))(conv7)
        pool2_list.append(pool2)

        merge_conv2_conv3 = concatenate([conv2, conv3], axis = -1)
        merge_list.append(merge_conv2_conv3)

        merge_conv5_conv6 = concatenate([conv5, conv6], axis = -1)
        merge_list.append(merge_conv5_conv6)

    merge_pool2 = concatenate(pool2_list, axis = 2)
    merge_list.append(merge_pool2)

    x = concatenate(merge_list, axis = 2)
    x = Flatten()(x)
    dense1_ = Dense(128, activation='relu')
    dense1  = dense1_(x)
    x = Dropout(0.11)(dense1)
    dense2  = Dense(256, activation='relu')(x)
    x = Dropout(0.47)(dense2)
    dense3 = Dense(32, activation='relu')(x)
    pred_output = Dense(1, activation='sigmoid')(dense3)
    model = Model(input=input_list, output=[pred_output])

    return model


def data():
    X = list() #[H3K27ac (x, 100), DNase (x, 100), ...] 
    for track in tracks_list:
        filename=args.input_dir+"/encoded_" + track + "_" + args.cell + ".pickle"
        print("reading pickle file: %s" % filename)
        f = open(filename, 'rb')
        X_train, y = pickle.load(f)
        f.close()
        X.append(X_train)

    kf = sklearn.model_selection.KFold(n_splits=5, random_state=None, shuffle=False)
    for train_index, test_index in kf.split(y):
        x_train = list()
        x_test = list()
        for x in X:
            x_train.append(np.array([np.array(x[i]) for i in train_index]).reshape((-1, 1, 100, 1)))
            x_test.append(np.array([np.array(x[i]) for i in test_index]).reshape((-1, 1, 100, 1)))
        y_train = np.array([np.array(y[i]) for i in train_index]).reshape(-1,1)
        y_test = np.array([np.array(y[i]) for i in test_index]).reshape(-1,1)
        break
    print(x_train[0].shape)
    print(x_test[0].shape)
    print(y_train.shape)
    print(y_test.shape)
    return x_train, y_train, x_test, y_test

if __name__ == '__main__':
    #parsing command line arguments
    parser = argparse.ArgumentParser(description='enter the tracks and cell line for CNN')
    parser.add_argument('-t', '--tracks', type=str, help='track of context information of length 100')
    parser.add_argument('-c', '--cell', type=str, help='Name of the cell to investigate')
    parser.add_argument('-i', '--input_dir', type=str, help='pickle file input directory')
    parser.add_argument('-n', '--name', type=str, help='output file root name')
    args = parser.parse_args()
    tracks_list = args.tracks.split(',')

    #structing input output name stems
    track_name = '+'.join(tracks_list)
    output_name = 'classifier.se.'+args.name+"."+track_name+'.'+args.cell+".histogram"

    X_train, Y_train, X_test, Y_test = data()

    model = load_model()

    adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=8e-4)

    os.system("mkdir -p ./tensorboard_log/"+output_name)
    datetime_str = ('{date:%Y-%m-%d-%H:%M:%S}'.format(date=datetime.now()))
    tensorboard = TensorBoard(log_dir='./tensorboard_log/'+output_name+"/"+datetime_str, 
        histogram_freq=1, batch_size=1)

    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy', auroc, auprc, f1_m, recall_m, precision_m])

    result = model.fit(X_train, Y_train,
              batch_size=1,
              epochs=30,
              validation_split=0.25,
              callbacks=[tensorboard])
