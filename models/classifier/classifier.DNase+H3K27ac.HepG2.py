import numpy as np
import string
import random
import os
os.environ["PATH"] += os.pathsep + '/gpfs/ysm/project/zc264/conda_envs/old_keras_gpu/lib/python3.6/site-packages/graphviz'
from sklearn.model_selection import train_test_split
import pickle
import math

import tensorflow as tf
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, SpatialDropout2D, BatchNormalization, LSTM, concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras.utils import Sequence, plot_model
from keras.constraints import unit_norm
from keras import regularizers
from keras.callbacks import EarlyStopping
from matplotlib import pyplot as plt

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
        fig, axs = plt.subplots(fig_dim,fig_dim, figsize=(20,20))
        fig.subplots_adjust(hspace = .5, wspace=.001)
        axs = axs.ravel()
        for i in range(int(W.shape[0])):
            axs[i].imshow(W[i,:,:])
            axs[i].set_title(str(i))
        fig.savefig('./figures/classifier.H3K27ac+DNase.HepG2.conv_layer.png')

es = EarlyStopping(monitor='val_auroc', mode='max', min_delta=0.0001, verbose=1, patience=10)

def auroc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred, curve="ROC")[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc

def auprc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred, curve='PR')[1]
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

def get_DNase(filename):
    print("reading pickle file: %s" % filename)
    f = open(filename, 'rb')
    X_train_dnase, y_train_dnase, X_test_dnase, y_test_dnase = pickle.load(f)
    f.close()

    return X_train_dnase, y_train_dnase, X_test_dnase, y_test_dnase

X_train_dnase, y_train_dnase, X_test_dnase, y_test_dnase = get_DNase("encoded_DNase_HepG2.pickle")
X_train_H3K27ac, y_train_H3K27ac, X_test_H3K27ac, y_test_H3K27ac = get_DNase("encoded_H3K27ac_HepG2.pickle")

# print(len(X_train_dnase), len(X_train_dnase[0]))
# print(y_train_dnase[:100])
# print(len(X_test_dnase), len(X_test_dnase[0]))

class MY_Generator(Sequence):

    def __init__(self, X_dnase, y_dnase, X_H3K27ac, y_H3K27ac):
        self.X_dnase = X_dnase
        self.y_dnase = y_dnase
        self.X_H3K27ac = X_H3K27ac
        self.y_H3K27ac = y_H3K27ac

    def __len__(self):
        x_len = len(self.X_dnase)
        # sprint(x_len)
        return x_len

    def __getitem__(self, idx):
        # print(idx)

        batch_x_dnase = self.X_dnase[idx]
        batch_x_flattened_dnase = np.array(batch_x_dnase)
        batch_x_flipped_dnase = batch_x_flattened_dnase.reshape((1, 1, 100, 1), order='F')

        batch_x_H3K27ac = self.X_H3K27ac[idx]
        batch_x_flattened_H3K27ac = np.array(batch_x_H3K27ac)
        batch_x_flipped_H3K27ac = batch_x_flattened_H3K27ac.reshape((1, 1, 100, 1), order='F')

        batch_y_dnase = np.array(self.y_dnase[idx]).reshape(1,1)

        return [batch_x_flipped_H3K27ac, batch_x_flipped_dnase], batch_y_dnase

train_generator = MY_Generator(X_train_dnase, y_train_dnase, X_train_H3K27ac, y_train_H3K27ac)
valid_generator = MY_Generator(X_test_dnase, y_test_dnase, X_test_H3K27ac, y_test_H3K27ac)
predict_gen = MY_Generator(X_test_dnase, y_test_dnase, X_test_H3K27ac, y_test_H3K27ac)
evaluate_gen = MY_Generator(X_test_dnase, y_test_dnase, X_test_H3K27ac, y_test_H3K27ac)

input_H3K27ac   = Input(shape=(1, 100, 1))
input_dnase = Input(shape=(1, 100, 1))
H3K27ac_conv1_ = Conv2D(128, (1, 10), activation='relu',padding='valid')
H3K27ac_conv1  = H3K27ac_conv1_(input_H3K27ac)
H3K27ac_conv2_ = Conv2D(64, (1, 1), activation='relu',padding='same')
H3K27ac_conv2  = H3K27ac_conv2_(H3K27ac_conv1)
H3K27ac_conv3_ = Conv2D(64, (1, 3), activation='relu',padding='same')
H3K27ac_conv3  = H3K27ac_conv3_(H3K27ac_conv2)
H3K27ac_conv4_ = Conv2D(128, (1, 1), activation='relu',padding='same')
H3K27ac_conv4  = H3K27ac_conv4_(H3K27ac_conv3)
H3K27ac_pool1  = MaxPooling2D(pool_size=(1, 2))(H3K27ac_conv4)
H3K27ac_conv5_ = Conv2D(64, (1, 3), activation='relu',padding='same')
H3K27ac_conv5  = H3K27ac_conv5_(H3K27ac_pool1)
H3K27ac_conv6_ = Conv2D(64, (1, 3), activation='relu',padding='same')
H3K27ac_conv6  = H3K27ac_conv6_(H3K27ac_conv5)
H3K27ac_pool2  = MaxPooling2D(pool_size=(1, 2))(H3K27ac_conv6)
dnase_conv1_ = Conv2D(128, (1, 10), activation='relu',padding='valid')
dnase_conv1  = dnase_conv1_(input_dnase)
dnase_conv2_ = Conv2D(64, (1, 1), activation='relu',padding='same')
dnase_conv2  = dnase_conv2_(dnase_conv1)
dnase_conv3_ = Conv2D(64, (1, 3), activation='relu',padding='same')
dnase_conv3  = dnase_conv3_(dnase_conv2)
dnase_conv4_ = Conv2D(128, (1, 1), activation='relu',padding='same')
dnase_conv4  = dnase_conv4_(dnase_conv3)
dnase_pool1  = MaxPooling2D(pool_size=(1, 2))(dnase_conv4)
dnase_conv5_ = Conv2D(64, (1, 3), activation='relu',padding='same')
dnase_conv5  = dnase_conv5_(dnase_pool1)
dnase_conv6_ = Conv2D(64, (1, 3), activation='relu',padding='same')
dnase_conv6  = dnase_conv6_(dnase_conv5)
dnase_pool2  = MaxPooling2D(pool_size=(1, 2))(dnase_conv6)
merge_H3K27ac_conv2_conv3 = concatenate([H3K27ac_conv2, H3K27ac_conv3], axis = -1)
merge_H3K27ac_conv5_conv6 = concatenate([H3K27ac_conv5, H3K27ac_conv6], axis = -1)
merge_dnase_conv2_conv3 = concatenate([dnase_conv2, dnase_conv3], axis = -1)
merge_dnase_conv5_conv6 = concatenate([dnase_conv5, dnase_conv6], axis = -1)
merge_pool2 = concatenate([H3K27ac_pool2, dnase_pool2], axis = -1)
x = concatenate([H3K27ac_conv1, merge_H3K27ac_conv2_conv3, merge_H3K27ac_conv5_conv6, merge_pool2, merge_dnase_conv5_conv6, merge_dnase_conv2_conv3, dnase_conv1], axis = 2)
x = Flatten()(x)
dense1_ = Dense(512, activation='relu')
dense1  = dense1_(x)
dense2  = Dense(256, activation='relu')(dense1)
x = Dropout(0.5)(dense2)
dense3  = Dense(128, activation='relu')(x)
pred_output = Dense(1, activation='sigmoid')(dense3)
model = Model(input=[input_H3K27ac, input_dnase], output=[pred_output])

# compile the keras model
adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-6)
model.compile(loss='binary_crossentropy', optimizer=adam, 
    metrics=['accuracy', auroc, auprc, f1_m, recall_m, precision_m])

print(model.summary())

history = model.fit_generator(train_generator, epochs=200, validation_data=valid_generator, 
    shuffle=False, max_queue_size=20, use_multiprocessing=True, workers=4, callbacks=[es])

y_pred = model.predict_generator(predict_gen).ravel()
#print(y_pred[:], y_test)
#y_pred = np.rint(y_pred)
y_test = np.array(y_test_H3K27ac)

accuracy_s = sklearn.metrics.accuracy_score(y_test, np.rint(y_pred))

# plot accuracy over time
plt.figure()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('./figures/classifier.H3K27ac+DNase.HepG2.accuracy.png')

# plot loss over time
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model binary entropy loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('./figures/classifier.H3K27ac+DNase.HepG2.loss.png')

# auroc over time
plt.figure()
plt.plot(history.history['auroc'])
plt.plot(history.history['val_auroc'])
plt.title('model auROC')
plt.ylabel('auroc')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('./figures/classifier.H3K27ac+DNase.HepG2.auROC.png')

# auprc over time
plt.figure()
plt.plot(history.history['auprc'])
plt.plot(history.history['val_auprc'])
plt.title('model auPRC')
plt.ylabel('auprc')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('./figures/classifier.H3K27ac+DNase.HepG2.auPRC.png')

# ROC in test set
plt.figure()
fpr_keras, tpr_keras, thresholds_keras = sklearn.metrics.roc_curve(y_test, y_pred)
auroc_s = sklearn.metrics.auc(fpr_keras, tpr_keras)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auroc_s))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.savefig('./figures/classifier.H3K27ac+DNase.HepG2.ROC.png')

# PRC in test set
plt.figure()
precision_keras, recall_keras, thresholds_keras = sklearn.metrics.precision_recall_curve(y_test, y_pred)
auprc_s = sklearn.metrics.auc(recall_keras, precision_keras)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(recall_keras, precision_keras, label='Keras (area = {:.3f})'.format(auprc_s))
plt.xlabel('Precision')
plt.ylabel('Recall')
plt.title('PR curve')
plt.legend(loc='best')
plt.savefig('./figures/classifier.H3K27ac+DNase.HepG2.PR.png')

print(accuracy_s, auroc_s, auprc_s)

# plot model
plot_model(model, to_file='./figures/classifier.H3K27ac+DNase.HepG2.model.png')

# plot certain dnase layers
plot_conv_weights(model, "conv2d_3")