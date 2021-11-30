import numpy as np
import string
import random
import os
from sklearn.model_selection import train_test_split
import pickle
import argparse
import math
from datetime import datetime

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
from keras.callbacks import EarlyStopping, Callback, TensorBoard, ReduceLROnPlateau
from matplotlib import pyplot as plt
import keras_metrics as km

import sklearn


# plot convolution layers
def plot_conv_weights(model, layer):
    plt.figure()
    W = model.get_layer(name=layer).get_weights()[0]
    if len(W.shape) == 4:
        W = np.squeeze(W, axis=0)
        print(W.shape)
        #W = W.reshape((W.shape[0], W.shape[1], W.shape[2]*W.shape[3])) 
        fig_dim = int(math.ceil(math.sqrt(W.shape[0])))
        fig, axs = plt.subplots(fig_dim,fig_dim, figsize=(fig_dim*4,fig_dim*4))
        fig.subplots_adjust(hspace = .5, wspace=.001)
        axs = axs.ravel()
        for i in range(int(W.shape[0])):
            axs[i].imshow(W[i,:,:])
            axs[i].set_title(str(i))
        fig.savefig(figure_output_name+'.'+layer+'_layer.png')


def SqueezeExcite(tensor, ratio=16):
    nb_channel = K.int_shape(tensor)[-1]

    x = GlobalAveragePooling2D()(tensor)
    x = Dense(nb_channel // ratio, activation='relu')(x)
    x = Dense(nb_channel, activation='sigmoid')(x)

    x = Multiply()([tensor, x])
    return x

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

def get_track(filename):
    print("reading pickle file: %s" % filename)
    f = open(filename, 'rb')
    X_train, y_train = pickle.load(f)
    # print(len(X_train), len(y_train))
    # for i in range(30):
    #     print(len(X_train[i]))
    f.close()

    return X_train, y_train

def get_data(tracks_list):
    track_X_list = list() #[H3K27ac (x, 100), DNase (x, 100), ...] 
    for track in tracks_list:
        filename=args.input_dir+"/encoded_" + track + "_" + args.cell + ".pickle"
        X_get, track_y = get_track(filename)
        track_X_list.append(X_get)

    return track_X_list, track_y #assuming all the track labels are sorted the same way

class LearningRatePrinter(Callback):
    def on_epoch_begin(self, epoch, logs=None):
        lr = self.model.optimizer.lr
        decay = self.model.optimizer.decay
        iterations = self.model.optimizer.iterations
        lr_with_decay = lr / (1. + decay * K.cast(iterations, K.dtype(decay)))
        print("lr: " + str(K.eval(lr_with_decay)))

def load_model(tracks_list):
    K.clear_session()
    pool2_list = []
    merge_list = []

    input_size = Input(shape=(3, 100, 1))
    conv1_ = Activation(activation='relu')(Conv2D(128, (3, 10), padding='same')(input_size))
    conv1  = SqueezeExcite(conv1_)
    merge_list.append(conv1)

    conv2_ = Activation(activation='relu')(Conv2D(64, (3, 1), padding='same')(conv1))
    conv2  = SqueezeExcite(conv2_)
    conv3_ = Activation(activation='relu')(Conv2D(64, (3, 3), padding='same')(conv2))
    conv3  = SqueezeExcite(conv3_)
    conv4_ = Activation(activation='relu')(Conv2D(128, (3, 1), padding='same')(conv3))
    conv4  = SqueezeExcite(conv4_)
    pool1  = MaxPooling2D(pool_size=(1, 2))(conv4)
    conv5_ = Activation(activation='relu')(Conv2D(64, (3, 3), padding='same')(pool1))
    conv5  = SqueezeExcite(conv5_)
    conv6_ = Activation(activation='relu')(Conv2D(64, (3, 3), padding='same')(conv5))
    conv6  = SqueezeExcite(conv6_)
    conv7_ = Activation(activation='relu')(Conv2D(128, (3, 1), padding='same')(conv6))
    conv7  = SqueezeExcite(conv7_)

    pool2  = MaxPooling2D(pool_size=(1, 2))(conv7)
    pool2_list.append(pool2)

    merge_conv2_conv3 = concatenate([conv2, conv3], axis = -1)
    merge_list.append(merge_conv2_conv3)

    merge_conv5_conv6 = concatenate([conv5, conv6], axis = -1)
    merge_list.append(merge_conv5_conv6)

    merge_list.extend(pool2_list)

    x = concatenate(merge_list, axis = 2)
    x = Flatten()(x)
    dense1_ = Dense(128, activation='relu')
    dense1  = dense1_(x)
    x = Dropout(0.11)(dense1)
    dense2  = Dense(256, activation='relu')(x)
    x = Dropout(0.47)(dense2)
    dense3 = Dense(32, activation='relu')(x)
    pred_output = Dense(1, activation='sigmoid')(dense3)
    model = Model(input=[input_size], output=[pred_output])

    return model

class track_generator(Sequence):

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        x_len = len(self.X[0])
        # sprint(x_len)
        return x_len

    def __getitem__(self, idx):
        # print(idx)

        # batch_x = self.X[idx]
        # batch_x_flattened = np.array(batch_x)
        # batch_x_flipped = batch_x_flattened.reshape((1, 1, 100, 1), order='F')
        # print(idx)

        X_extract = [np.array(x[idx]).reshape((1, 1, 100, 1)) for x in self.X]
        X_extract[0].shape
        X_bunched = np.array(X_extract).reshape((1, 3, 100, 1))
        X_bunched.shape

        batch_y = np.array(self.y[idx]).reshape(1,1)

        return X_bunched, batch_y

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
    output_name = 'classifier.se.'+args.name+"."+track_name+'.'+args.cell
    figure_output_name = './figures/' + output_name
    print(figure_output_name)

    #get all the data from all the tracks
    #X is [[DHS], [H3K27ac], ... (tracks)]
    #y is [1, 0, 1 ...(labels)]
    X, y = get_data(tracks_list)

    #kfold division of the data
    kf = sklearn.model_selection.KFold(n_splits=10, random_state=None, shuffle=False)

    #iterate over each fold of data
    for train_index, test_index in kf.split(y):
        x_train = list()
        x_test = list()
        for x in X:
            x_train.append([x[i] for i in train_index])
            x_test.append([x[i] for i in test_index])
        y_train = [y[i] for i in train_index]
        y_test = [y[i] for i in test_index]

        # construct generators for the data sets
        train_generator = track_generator(x_train, y_train)
        valid_generator = track_generator(x_test, y_test)

        # construct the model
        model = load_model(tracks_list)
        adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=9e-5)

        model.compile(loss='binary_crossentropy', optimizer=adam, 
            metrics=['accuracy', auroc, auprc, f1_m, recall_m, precision_m])

        #fit the model
        model.fit_generator(train_generator, epochs=30, validation_data=valid_generator, 
            shuffle=False, max_queue_size=20, use_multiprocessing=True, workers=4)#, 
            #callbacks=[lr_printer, mcheckpoint, tensorboard, reduce_lr]))

        #save the model and the weights in case if the model doesn't work
        model.save_weights('./model/'+output_name+'/' + output_name + '.weights.h5')
        model.save('./model/'+output_name+'/' + output_name + '.h5')

        break
