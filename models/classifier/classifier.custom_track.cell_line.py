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
    print(len(X_train, y_train))
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

def SqueezeExcite(tensor, ratio=16):
    nb_channel = K.int_shape(tensor)[-1]

    x = GlobalAveragePooling2D()(tensor)
    x = Dense(nb_channel // ratio, activation='relu')(x)
    x = Dense(nb_channel, activation='sigmoid')(x)

    x = Multiply()([tensor, x])
    return x


def load_model(tracks_list):
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

    if len(tracks_list) > 1:
        merge_pool2 = concatenate(pool2_list, axis = 2)
        merge_list.append(merge_pool2)
    else:
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
    model = Model(input=input_list, output=[pred_output])

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

        X_extract = [np.array(x[idx]).reshape((1, 1, 100, 1), order='F') for x in self.X]

        batch_y = np.array(self.y[idx]).reshape(1,1)

        return X_extract, batch_y

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
        x_train = list()
        x_test = list()
        for x in X:
            x_train.append([x[i] for i in train_index])
            x_test.append([x[i] for i in test_index])
        y_train = [y[i] for i in train_index]
        y_test = [y[i] for i in test_index]
        # print(train_index, test_index)
        # print(len(train_index), len(test_index))
        # print(len(x_train[0]), len(y_train))


        # construct generators for the data sets
        train_generator = track_generator(x_train, y_train)
        valid_generator = track_generator(x_test, y_test)
        predict_gen = track_generator(x_test, y_test)

        # construct the model
        model = load_model(tracks_list)
        adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=8e-4)
        #es = EarlyStopping(monitor='val_auroc', mode='max', min_delta=0.0001, verbose=1, patience=10)
        
        os.system("mkdir -p ./tensorboard_log/"+output_name)
        os.system("mkdir -p ./model/"+output_name)
        datetime_str = ('{date:%Y-%m-%d-%H:%M:%S}'.format(date=datetime.now()))
        lr_printer = LearningRatePrinter()
        mcheckpoint = ModelCheckpoint('./model/'+output_name+'/'+str(kskip) +'CV.{epoch:02d}-{val_loss:.2f}.h5', 
            monitor='val_loss', save_best_only=True, verbose=0, mode='min', period=5)
        tensorboard = TensorBoard(log_dir='./tensorboard_log/'+output_name+"/"+datetime_str, 
            histogram_freq=0, write_graph=True, write_images=True, write_grads=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                  patience=5, min_lr=0.000001)

        model.compile(loss='binary_crossentropy', optimizer=adam, 
            metrics=['accuracy', auroc, auprc, f1_m, recall_m, precision_m])

        #fit the model
        history_list.append(model.fit_generator(train_generator, epochs=30, validation_data=valid_generator, 
            shuffle=False, max_queue_size=20, use_multiprocessing=True, workers=4, 
            callbacks=[lr_printer, mcheckpoint, tensorboard, reduce_lr]))

        #save the model and the weights in case if the model doesn't work
        model.save_weights('./model/'+output_name+'/' + str(kskip) + '.weights.h5')
        model.save('./model/'+output_name+'/' + str(kskip) + '.h5')

        # predict the results
        y_pred = model.predict_generator(predict_gen).ravel()
        y_test = np.array(y_test)
        y_pred_list.append(y_pred)
        y_test_list.append(y_test)

        accuracy_s = sklearn.metrics.accuracy_score(y_test, np.rint(y_pred))
        accuracy_list.append(accuracy_s)

        #plot the model and the weights
        if display_model == True:
            print(model.summary())
            plot_model(model, to_file=figure_output_name+'.model.png')
            # plot_conv_weights(model, "conv2d_3")
            # plot_conv_weights(model, "conv2d_9")
            display_model = False

        #iterate k fold counter
        kskip = kskip + 1

        #delete the model so the variable is cleared
        del model
        # if kskip == 2:
        #     break


    # save intermediate results
    for his_metrics in ['acc', 'val_acc', 'loss', 'val_loss', 'auroc', 'val_auroc', 'auprc', 'val_auprc']:
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

    history_val_acc = np.array([np.array(h.history['val_acc']) for h in history_list])
    mean_history_val_acc = np.mean(history_val_acc, axis=0)

    plt.plot(mean_history_acc)
    plt.plot(mean_history_val_acc)
    plt.title('average accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(figure_output_name+'.accuracy.png')

    # plot loss over time
    plt.figure()
    history_loss = np.array([np.array(h.history['loss']) for h in history_list])
    mean_history_loss = np.mean(history_loss, axis=0)

    history_val_loss = np.array([np.array(h.history['val_loss']) for h in history_list])
    mean_history_val_loss = np.mean(history_val_loss, axis=0)

    plt.plot(mean_history_loss)
    plt.plot(mean_history_val_loss)
    plt.title('average loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(figure_output_name+'.loss.png')

    # auroc over time
    plt.figure()
    history_auroc = np.array([np.array(h.history['auroc']) for h in history_list])
    mean_history_auroc = np.mean(history_auroc, axis=0)

    history_val_auroc = np.array([np.array(h.history['val_auroc']) for h in history_list])
    mean_history_val_auroc = np.mean(history_val_auroc, axis=0)

    plt.plot(mean_history_auroc)
    plt.plot(mean_history_val_auroc)
    plt.title('averge auROC')
    plt.ylabel('auroc')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(figure_output_name+'.auROC.png')

    # auprc over time
    plt.figure()
    history_auprc = np.array([np.array(h.history['auprc']) for h in history_list])
    mean_history_auprc = np.mean(history_auprc, axis=0)

    history_val_auprc = np.array([np.array(h.history['val_auprc']) for h in history_list])
    mean_history_val_auprc = np.mean(history_val_auprc, axis=0)

    plt.plot(mean_history_auprc)
    plt.plot(mean_history_val_auprc)
    plt.title('averge auPRC')
    plt.ylabel('auprc')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(figure_output_name+'.auPRC.png')


    # ROC in test set
    plt.figure(figsize=(5, 5))
    base_fpr = np.linspace(0, 1, 101)
    tpr_list = []
    for i in range(len(y_test_list)):
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_test_list[i], y_pred_list[i])
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
    plt.plot(base_fpr, mean_tpr, 'b', label='Keras (area = {:.3f})'.format(sklearn.metrics.auc(np.linspace(0, 1, mean_tpr.shape[0]), mean_tpr)))
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
    for i in range(len(y_test_list)):
        recall, precision, thresholds = sklearn.metrics.precision_recall_curve(y_test_list[i], y_pred_list[i])
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
    plt.plot(base_recall, mean_precision, 'b', label='Keras (area = {:.3f})'.format(sklearn.metrics.auc(np.linspace(0, 1, mean_precision.shape[0]), mean_precision)))
    plt.fill_between(base_recall, precisions_lower, precisions_upper, color='grey', alpha=0.3)
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.title('PRC curve')
    plt.legend(loc='best')
    plt.axes().set_aspect('equal', 'datalim')
    plt.savefig(figure_output_name+'.PRC.png')
