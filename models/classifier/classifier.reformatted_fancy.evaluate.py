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
from keras.applications.vgg19 import VGG19
from keras.applications.xception import Xception
from keras.applications.inception_v3 import InceptionV3
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, SpatialDropout2D, BatchNormalization, LSTM, concatenate, Activation, GlobalAveragePooling2D, Multiply
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras.utils import Sequence, plot_model
from keras.constraints import unit_norm
from keras import regularizers
from keras.callbacks import Callback, TensorBoard, ReduceLROnPlateau
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
    print("reading file: %s" % filename)
    f = open(filename, "rb")
    X_train = np.loadtxt(f, delimiter="\t")
    # print(len(X_train), len(y_train))
    # for i in range(30):
    #     print(len(X_train[i]))
    f.close()

    return X_train

def new_get_data(tracks_list, labels):
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
        x_reform = np.array(x_reform)
        x_reform = np.resize(x_reform, (75, 75, 3))
        new_x_list.append(x_reform)
    new_x_list = np.array(new_x_list)

    print(new_x_list.shape)

    return new_x_list, y_train

class LearningRatePrinter(Callback):
    def on_epoch_begin(self, epoch, logs=None):
        lr = self.model.optimizer.lr
        decay = self.model.optimizer.decay
        iterations = self.model.optimizer.iterations
        lr_with_decay = lr / (1. + decay * K.cast(iterations, K.dtype(decay)))
        print("lr: " + str(K.eval(lr_with_decay)))


def load_model():
    K.clear_session()
    
    base_model = Xception(include_top=False, weights='imagenet', input_shape=(75, 75, 3), pooling="max")
    base_model.summary()
    x = base_model.output
    dense1_ = Dense(512, activation='relu')
    dense1  = dense1_(x)
    x = Dropout(0.2)(dense1)
    dense2  = Dense(256, activation='relu')(x)
    x = Dropout(0.2)(dense2)
    dense3 = Dense(128, activation='relu')(x)
    pred_output = Dense(1, activation='sigmoid')(dense3)
    model = Model(inputs=base_model.input, outputs=[pred_output])
    model.summary()
    return model


    # input_size = Input(shape=(3, 200, 1))
    # conv1_ = Conv2D(128, (3, 10), padding='same',activation='relu')(input_size)
    # conv2_ = Conv2D(64, (3, 1), padding='same',activation='relu')(conv1_)
    # conv3_ = Conv2D(64, (3, 3), padding='same',activation='relu')(conv2_)
    # conv4_ = Conv2D(128, (3, 1), padding='same',activation='relu')(conv3_)
    # pool1  = MaxPooling2D(pool_size=(1, 2))(conv4_)
    # conv5_ = Conv2D(64, (3, 3), padding='same',activation='relu')(pool1)
    # conv6_ = Conv2D(64, (3, 3), padding='same',activation='relu')(conv5_)
    # conv7_ = Conv2D(128, (3, 1), padding='same',activation='relu')(conv6_)
    # pool2  = MaxPooling2D(pool_size=(1, 2))(conv7_)

    # x = Flatten()(pool2)
    # dense1_ = Dense(512, activation='relu')
    # dense1  = dense1_(x)
    # x = Dropout(0.35)(dense1)
    # dense2  = Dense(256, activation='relu')(x)
    # x = Dropout(0.35)(dense2)
    # dense3 = Dense(128, activation='relu')(x)
    # pred_output = Dense(1, activation='sigmoid')(dense3)
    # model = Model(input=[input_size], output=[pred_output])
    # model.summary()

    return model


if __name__ == '__main__':

    #parsing command line arguments
    parser = argparse.ArgumentParser(description='enter the tracks and cell line for CNN')
    parser.add_argument('-t', '--tracks', type=str, help='track of context information of length 200')
    parser.add_argument('-l', '--labels', type=str, help='label information on the CNN')
    parser.add_argument('-n', '--name', type=str, help='output file root name')
    args = parser.parse_args()
    tracks_list = args.tracks.split(',')

    #structing input output name stems
    output_name = 'classifier.reformatted_fancy.'+args.name
    figure_output_name = './figures/' + output_name
    print(figure_output_name)

    #X is a list of numpy arrays, Y is the labels
    X, y = new_get_data(tracks_list, args.labels)

    # construct the model
    model = load_model()

    adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=9e-5)
    model.compile(loss='binary_crossentropy', optimizer=adam, 
        metrics=['accuracy', auroc, auprc, f1_m, recall_m, precision_m])

    callbacks = [EarlyStopping(monitor='val_loss', patience=3,restore_best_weights=True)]

    #train the model
    history = model.fit(X, y,
                callbacks=callbacks,
                batch_size=1,
                epochs=150,
                validation_split=0.1)
    #save the model and the weights in case if the model doesn't work
    os.system("mkdir -p ./model/"+output_name)
    model.save_weights('./model/'+output_name+'/' + output_name + '.weights.h5')
    model.save('./model/'+output_name+'/' + output_name + '.h5')


    # plot accuracy over time 
    plt.figure()
    plt.plot(history.history['acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    plt.savefig(figure_output_name+'.accuracy.png')

    # plot loss over time
    plt.figure()
    plt.plot(history.history['loss'])
    plt.title('model binary entropy loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    plt.savefig(figure_output_name+'.loss.png')

    # auroc over time
    plt.figure()
    plt.plot(history.history['auroc'])
    plt.title('model auROC')
    plt.ylabel('auroc')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    plt.savefig(figure_output_name+'.auROC.png')

    # auprc over time
    plt.figure()
    plt.plot(history.history['auprc'])
    plt.title('model auPRC')
    plt.ylabel('auprc')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    plt.savefig(figure_output_name+'.auPRC.png')

