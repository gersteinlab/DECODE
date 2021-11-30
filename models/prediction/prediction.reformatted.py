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
from keras.models import Sequential, Model, load_model
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

def all_data(input_dir, cell):

    # obtain all of the data in the pickle files

    X = list() #[H3K27ac (x, 100), DNase (x, 100), ...] 
    for track in tracks_list:
        filename=args.input_dir+"/encoded_" + track + "_" + cell + ".sorted.pickle"
        print("reading pickle file: %s" % filename)
        f = open(filename, 'rb')
        X_train= pickle.load(f)
        f.close()
        X.append(X_train)

    x_train = list()
    for x in X:
        x_train.append(np.array([np.array(x[i]) for i in range(len(x))]).reshape((-1, 1, 100, 1)))

    print(x_train[0].shape)

    return x_train

class LearningRatePrinter(Callback):
    def on_epoch_begin(self, epoch, logs=None):
        lr = self.model.optimizer.lr
        decay = self.model.optimizer.decay
        iterations = self.model.optimizer.iterations
        lr_with_decay = lr / (1. + decay * K.cast(iterations, K.dtype(decay)))
        print("lr: " + str(K.eval(lr_with_decay)))

def get_track(filename):
    print("reading file: %s" % filename)
    f = open(filename, "rb")
    X_train = np.loadtxt(f, delimiter="\t")
    # print(len(X_train), len(y_train))
    # for i in range(30):
    #     print(len(X_train[i]))
    f.close()

    return X_train

def get_XY(tracks_list, labels):
    track_X_list = list() #[H3K27ac (x, 200), DNase (x, 200), ...] 
    for track in tracks_list:
        X_get = get_track(track)
        track_X_list.append(X_get)

    f=open(labels, "rb")
    y_train = np.loadtxt(f, delimiter="\t", usecols = 0)
    y_train = y_train.reshape((len(y_train), 1))
    print(y_train.shape[0])
    print(y_train[0:10])


    for i in range(len(track_X_list)):
        if track_X_list[i].shape[0] != y_train.shape[0]:
            print("%s input track shape does not correspond to label" % (str(i)))
            print(track_X_list[i].shape[0])
            print(y_train.shape[0])
            exit(1)

    new_x_list = list()
    for i in range(len(y_train)):
        x_reform = list()
        for x in track_X_list:
            x_reform.append(x[i])
        new_x_list.append(np.array(x_reform))
    new_x_list = np.expand_dims(np.array(new_x_list), axis=3)

    print(new_x_list.shape)

    return new_x_list, y_train

def get_X(tracks_list):
    track_X_list = list() #[H3K27ac (x, 200), DNase (x, 200), ...] 
    for track in tracks_list:
        X_get = get_track(track)
        track_X_list.append(X_get)

    new_x_list = list()
    for i in range(track_X_list[0].shape[0]):
        x_reform = list()
        for x in track_X_list:
            x_reform.append(x[i])
        new_x_list.append(np.array(x_reform))
    new_x_list = np.expand_dims(np.array(new_x_list), axis=3)

    print(new_x_list.shape)

    return new_x_list

if __name__ == '__main__':

    #parsing command line arguments
    parser = argparse.ArgumentParser(description='enter the tracks and cell line for CNN')
    parser.add_argument('-i', '--tracks', type=str, help='track of context information of length 200')
    parser.add_argument('-o', '--output_dir', type=str, help='output directory')
    parser.add_argument('-n', '--name', type=str, help='output file root name')
    parser.add_argument('-m', '--model', type=str, help='trained model')
    args = parser.parse_args()
    tracks_list = args.tracks.split(',')

    #constructing input output name stems
    track_name = '+'.join(tracks_list)
    output_name = args.name + ".VGG"

    #load trained model
    dependencies = {'auroc': auroc, 'auprc':auprc, 'f1_m':f1_m, 'recall_m':recall_m, 'precision_m':precision_m}
    model = load_model(args.model, custom_objects=dependencies)

    #get the prediction data
    X = get_X(tracks_list)

    #predict the output
    y_pred = model.predict(X)

    #reshape the output into 1d in the correct order
    y_pred.shape
    y_out = y_pred.reshape(-1)
    y_out.shape

    np.savetxt(args.output_dir+'/'+output_name+'.prediction.tsv', y_out, delimiter='\n')





