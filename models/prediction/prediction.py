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



if __name__ == '__main__':

    #parsing command line arguments
    parser = argparse.ArgumentParser(description='enter the tracks and cell line for CNN')
    parser.add_argument('-c', '--cell', type=str, help='name of the cell')
    parser.add_argument('-t', '--tracks', type=str, help='track of context information of length 200')
    parser.add_argument('-i', '--input_dir', type=str, help='input directory with tracks')
    parser.add_argument('-n', '--name', type=str, help='output file root name')
    parser.add_argument('-m', '--model', type=str, help='trained model')
    args = parser.parse_args()
    tracks_list = args.tracks.split(',')

    #constructing input output name stems
    track_name = '+'.join(tracks_list)
    output_name = 'predictor.se.'+args.name+"."+track_name+'.'+args.cell
    figure_output_name = './figures/' + output_name
    print(figure_output_name)

    #load trained model
    dependencies = {'auroc': auroc, 'auprc':auprc, 'f1_m':f1_m, 'recall_m':recall_m, 'precision_m':precision_m}
    model = load_model(args.model,custom_objects=dependencies)

    #get the prediction data (make into generator?)
    X = all_data(args.input_dir, args.cell)

    #predict the output
    y_pred = model.predict(X)

    #reshape the output into 1d in the correct order
    y_pred.shape
    y_out = y_pred.reshape(-1)
    y_out.shape

    #output the confidence score for each 1kb region
    np.savetxt('./pred_result/'+output_name+'.tsv', y_out, delimiter='\n')





