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

def get_track(filename):
    print("reading file: %s" % filename)
    f = open(filename, "rb")
    X_train = np.loadtxt(f, delimiter="\t")
    # print(len(X_train), len(y_train))
    # for i in range(30):
    #     print(len(X_train[i]))
    f.close()

    return X_train

def get_data(tracks_list, labels):
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

class LearningRatePrinter(Callback):
    def on_epoch_begin(self, epoch, logs=None):
        lr = self.model.optimizer.lr
        decay = self.model.optimizer.decay
        iterations = self.model.optimizer.iterations
        lr_with_decay = lr / (1. + decay * K.cast(iterations, K.dtype(decay)))
        print("lr: " + str(K.eval(lr_with_decay)))


def load_model():
    K.clear_session()
    #Create your own input format
    input_size = Input(shape=(3,200,1),name = 'track_input')

    x = Conv2D(64, (3, 3),
                  activation='relu',
                  padding='same',
                  name='block1_conv1')(input_size)
    x = Conv2D(64, (3, 3),
                  activation='relu',
                  padding='same',
                  name='block1_conv2')(x)
    x = MaxPooling2D(pool_size=(1, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3),
                  activation='relu',
                  padding='same',
                  name='block2_conv1')(x)
    x = Conv2D(128, (3, 3),
                  activation='relu',
                  padding='same',
                  name='block2_conv2')(x)
    x = MaxPooling2D(pool_size=(1, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3),
                  activation='relu',
                  padding='same',
                  name='block3_conv1')(x)
    x = Conv2D(256, (3, 3),
                  activation='relu',
                  padding='same',
                  name='block3_conv2')(x)
    x = Conv2D(256, (3, 3),
                  activation='relu',
                  padding='same',
                  name='block3_conv3')(x)
    x = Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv4')(x)
    x = MaxPooling2D(pool_size=(1, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv1')(x)
    x = Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv2')(x)
    x = Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv3')(x)
    x = Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv4')(x)
    x = MaxPooling2D(pool_size=(1, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv1')(x)
    x = Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv2')(x)
    x = Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv3')(x)
    x = Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv4')(x)
    x = MaxPooling2D(pool_size=(1, 2), name='block5_pool')(x)

    #Add the fully-connected layers 
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dense(1, activation='softmax', name='predictions')(x)

    #Create your own model 
    my_model = Model(input=input_size, output=x)
    my_model.summary()

    return my_model

def load_model_SENet():
    K.clear_session()
    pool2_list = []
    merge_list = []

    input_size = Input(shape=(3, 200, 1))
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
    model.summary()

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
    output_name = 'classifier.reformatted.cv.'+args.name
    figure_output_name = './figures/' + output_name
    print(figure_output_name)

    #X is a list of numpy arrays, Y is the labels
    X, y = get_data(tracks_list, args.labels)

    #kfold division of the data
    kf = sklearn.model_selection.KFold(n_splits=5, random_state=None, shuffle=False)

    #collect the output of the kfolds
    history_list = []
    y_pred_list = []
    y_test_list = []
    accuracy_list = []

    display_model = True #print model only once
    kskip = 0

    #iterate over each fold of data
    for train_index, test_index in kf.split(y):

        x_train = X[train_index]
        x_test = X[test_index]
        y_train = y[train_index]
        y_test = y[test_index]

        # construct the model
        model = load_model_SENet()

        adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=9e-5)
        model.compile(loss='binary_crossentropy', optimizer=adam, 
            metrics=['accuracy', auroc, auprc, f1_m, recall_m, precision_m])

        #train the model
        history_list.append(model.fit(x_train, y_train,
                    batch_size=1,
                    epochs=30,
                    validation_split=0.0))

        # predict the results
        y_pred = model.predict(x_test).ravel()
        y_pred_list.append(y_pred)
        y_test_list.append(y_test.ravel())

        accuracy_s = sklearn.metrics.accuracy_score(y_test, np.rint(y_pred))
        accuracy_list.append(accuracy_s)

        #plot the model and the weights
        if display_model == True:
            print(model.summary())
            plot_model(model, to_file=figure_output_name+'.model.png')
            display_model = False

        #iterate k fold counter
        kskip = kskip + 1

        #delete the model so the variable is cleared
        del model
        # if kskip == 2:
        #     break

    # save intermediate results
    for his_metrics in ['acc', 'loss', 'auroc', 'auprc']:
        temp_list = np.array([np.array(his.history[his_metrics]) for his in history_list])
        np.savetxt("./history/"+output_name+"."+his_metrics+".tsv", temp_list, delimiter="\t")

    print(len(y_test_list), len(y_pred_list))
    print(type(y_test_list), type(y_pred_list))

    y_test_out = []
    y_pred_out = []
    for j in range(len(y_test_list)):
        print(len(y_test_list[j]), len(y_pred_list[j]), type(y_test_list[j]))
        y_test_out.extend(y_test_list[j])
        y_pred_out.extend(y_pred_list[j])

    filehandler1 = open("./history/"+output_name+"."+"y.pickle",'wb')
    pickle.dump([y_test_list, y_pred_list], filehandler1)
    filehandler1.close()
    np.savetxt("./history/"+output_name+"."+"y_test.tsv", np.array(y_test_out), delimiter="\n")
    np.savetxt("./history/"+output_name+"."+"y_pred.tsv", np.array(y_pred_out), delimiter="\n")

    # plot accuracy over time
    plt.figure()
    history_acc = np.array([np.array(h.history['acc']) for h in history_list])
    mean_history_acc = np.mean(history_acc, axis=0)

    plt.plot(mean_history_acc, label='Keras (5cv_acc = {:.3f})'.format(np.mean(np.array(accuracy_list))))
    plt.title('training and final validation accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    plt.savefig(figure_output_name+'.accuracy.png')

    # plot loss over time
    plt.figure()
    history_loss = np.array([np.array(h.history['loss']) for h in history_list])
    mean_history_loss = np.mean(history_loss, axis=0)

    plt.plot(mean_history_loss)
    plt.title('training loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    plt.savefig(figure_output_name+'.loss.png')

    # auroc over time
    plt.figure()
    history_auroc = np.array([np.array(h.history['auroc']) for h in history_list])
    mean_history_auroc = np.mean(history_auroc, axis=0)

    plt.plot(mean_history_auroc)
    plt.title('training auROC')
    plt.ylabel('auroc')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    plt.savefig(figure_output_name+'.auROC.png')

    # auprc over time
    plt.figure()
    history_auprc = np.array([np.array(h.history['auprc']) for h in history_list])
    mean_history_auprc = np.mean(history_auprc, axis=0)

    plt.plot(mean_history_auprc)
    plt.title('training auPRC')
    plt.ylabel('auprc')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    plt.savefig(figure_output_name+'.auPRC.png')


    # ROC in test set
    plt.figure(figsize=(5, 5))
    base_fpr = np.linspace(0, 1, 101)
    tpr_list = []
    auroc_list = []
    for i in range(len(y_test_list)):
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_test_list[i], y_pred_list[i])
        auroc_list.append(sklearn.metrics.roc_auc_score(y_test_list[i], y_pred_list[i]))
        plt.plot(fpr, tpr, 'b', alpha=0.15)
        tpr = np.interp(base_fpr, fpr, tpr)
        tpr[0] = 0.0
        tpr_list.append(tpr)
    
    
    print(len(tpr_list), len(tpr_list[0]), len(tpr_list[1]))
    tpr_list = np.array(tpr_list)
    mean_tpr = np.mean(np.array(tpr_list), axis=0)
    tpr_std = tpr_list.std(axis=0)

    tprs_upper = np.minimum(mean_tpr + 2 * tpr_std, 1)
    tprs_lower = mean_tpr - 2 * tpr_std

    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(base_fpr, mean_tpr, 'b', label='Keras (area = {:.3f})'.format(np.mean(np.array(auroc_list))))
    plt.fill_between(base_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.3)
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.axes().set_aspect('equal', 'datalim')
    plt.savefig(figure_output_name+'.ROC.png')

    # PRC in test set
    plt.figure(figsize=(5, 5))
    base_recall = np.linspace(0, 1, 101)
    precision_list = []
    auprc_list = []
    for i in range(len(y_test_list)):
        recall, precision, thresholds = sklearn.metrics.precision_recall_curve(y_test_list[i], y_pred_list[i])
        auprc_list.append(sklearn.metrics.average_precision_score(y_test_list[i], y_pred_list[i]))
        plt.plot(recall, precision, 'b', alpha=0.15)
        precision = np.interp(base_recall, recall, precision)
        precision[0] = 1.0
        precision_list.append(precision)
        
    print(len(precision_list), len(precision_list[0]), len(precision_list[1]))
    precision_list = np.array(precision_list)
    mean_precision = np.mean(np.array(precision_list), axis=0)
    precision_std = precision_list.std(axis=0)

    precisions_upper = np.minimum(mean_precision + 2 * precision_std, 1)
    precisions_lower = mean_precision - 2 * precision_std

    plt.plot([0, 1], [1, 0], 'k--')
    plt.plot(base_recall, mean_precision, 'b', label='Keras (area = {:.3f})'.format(np.mean(np.array(auprc_list))))
    plt.fill_between(base_recall, precisions_lower, precisions_upper, color='grey', alpha=0.3)
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.title('PRC curve')
    plt.legend(loc='best')
    plt.axes().set_aspect('equal', 'datalim')
    plt.savefig(figure_output_name+'.PRC.png')

    print(np.mean(np.array(accuracy_list)), np.mean(np.array(auroc_list)), np.mean(np.array(auprc_list)))
