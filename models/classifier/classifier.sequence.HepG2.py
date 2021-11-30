import numpy as np
import string
import random
import os
from sklearn.model_selection import train_test_split
import pickle

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
from keras.utils import Sequence
from keras.constraints import unit_norm
from keras import regularizers
from matplotlib import pyplot as plt


import sklearn

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


def get_sequence(filename):
	print("reading pickle file: %s" % filename)
	f = open(filename, 'rb')
	X_train, y_train, X_test, y_test = pickle.load(f)
	f.close()

	return X_train, y_train, X_test, y_test

X_train, y_train, X_test, y_test = get_sequence("encoded_sequence_HepG2.pickle")
print(len(X_train), len(X_train[0]))
print(y_train[:100])
print(len(X_test), len(X_test[0]))

class MY_Generator(Sequence):

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        x_len = len(self.X)
        # sprint(x_len)
        return x_len

    def __getitem__(self, idx):
        # print(idx)

        batch_x = self.X[idx]
        batch_x_flattened = np.array([np.array(sublist) for sublist in batch_x])
        batch_x_flipped = batch_x_flattened.reshape((1, 4, 1000, 1), order='F')

        batch_y = np.array(self.y[idx]).reshape(1,1)

        return batch_x_flipped, batch_y

train_generator = MY_Generator(X_train, y_train)
valid_generator = MY_Generator(X_test, y_test)
predict_gen = MY_Generator(X_test, y_test)
evaluate_gen = MY_Generator(X_test, y_test)

#define the keras model
# model = Sequential()
# input_seq  = Input(shape=(4, 1000, 1))
# seq_conv1_ = Conv2D(128, (4, 8), activation='relu',padding='valid',dim_ordering='tf')
# seq_conv1  = seq_conv1_(input_seq)
# seq_conv2_ = Conv2D(64, (1, 1), activation='relu',padding='same')
# seq_conv2  = seq_conv2_(seq_conv1)
# seq_conv3_ = Conv2D(64, (1, 3), activation='relu',padding='same')
# seq_conv3  = seq_conv3_(seq_conv2)
# seq_conv4_ = Conv2D(128, (1, 1), activation='relu',padding='same')
# seq_conv4  = seq_conv4_(seq_conv3)
# seq_pool1  = MaxPooling2D(pool_size=(1, 2))(seq_conv4)
# seq_conv5_ = Conv2D(64, (1, 3), activation='relu',padding='same')
# seq_conv5  = seq_conv5_(seq_pool1)
# seq_conv6_ = Conv2D(64, (1, 3), activation='relu',padding='same')
# seq_conv6  = seq_conv6_(seq_conv5)
# #
# seq_conv7_ = Conv2D(128, (1, 1), activation='relu',padding='same')
# seq_conv7  = seq_conv7_(seq_conv6)
# #
# seq_pool2  = MaxPooling2D(pool_size=(1, 2))(seq_conv7)
# merge_seq_conv2_conv3 = concatenate([seq_conv2, seq_conv3], axis = -1)
# merge_seq_conv5_conv6 = concatenate([seq_conv5, seq_conv6], axis = -1)
# x = concatenate([seq_conv1, merge_seq_conv2_conv3, merge_seq_conv5_conv6, seq_pool2], axis = 2)
# x = Flatten()(x)
# dense1_ = Dense(512, activation='relu')
# dense1  = dense1_(x)
# dense2  = Dense(256, activation='relu')(dense1)
# x = Dropout(0.5)(dense2)
# dense3 = Dense(128, activation='relu')(x)
# pred_output = Dense(1, activation='sigmoid')(dense3)
# model = Model(input=[input_seq], output=[pred_output])

#define the keras model
model = Sequential()
input_seq  = Input(shape=(4, 1000, 1))
seq_conv1_ = Conv2D(128, (4, 10), strides=(4,10), activation='relu',padding='same')
seq_conv1  = seq_conv1_(input_seq)
seq_conv2_ = Conv2D(64, (1, 1), activation='relu',padding='same')
seq_conv2  = seq_conv2_(seq_conv1)
seq_conv3_ = Conv2D(64, (1, 3), activation='relu',padding='same')
seq_conv3  = seq_conv3_(seq_conv2)
seq_conv4_ = Conv2D(128, (1, 1), activation='relu',padding='same')
seq_conv4  = seq_conv4_(seq_conv3)
seq_pool1  = MaxPooling2D(pool_size=(1, 2))(seq_conv4)
seq_conv5_ = Conv2D(64, (1, 3), activation='relu',padding='same')
seq_conv5  = seq_conv5_(seq_pool1)
seq_conv6_ = Conv2D(64, (1, 3), activation='relu',padding='same')
seq_conv6  = seq_conv6_(seq_conv5)
#
seq_conv7_ = Conv2D(128, (1, 1), activation='relu',padding='same')
seq_conv7  = seq_conv7_(seq_conv6)
#
seq_pool2  = MaxPooling2D(pool_size=(1, 2))(seq_conv7)
merge_seq_conv2_conv3 = concatenate([seq_conv2, seq_conv3], axis = -1)
merge_seq_conv5_conv6 = concatenate([seq_conv5, seq_conv6], axis = -1)
x = concatenate([seq_conv1, merge_seq_conv2_conv3, merge_seq_conv5_conv6, seq_pool2], axis = 2)
x = Flatten()(x)
dense1_ = Dense(24, activation='relu')
dense1  = dense1_(x)
dense2  = Dense(12, activation='relu')(dense1)
x = Dropout(0.5)(dense2)
dense3 = Dense(6, activation='relu')(x)
pred_output = Dense(1, activation='sigmoid')(dense3)
model = Model(input=[input_seq], output=[pred_output])

# compile the keras model
adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-6)
model.compile(loss='binary_crossentropy', optimizer=adam, 
	metrics=['accuracy', auroc, auprc, f1_m, recall_m, precision_m])

print(model.summary())

history = model.fit_generator(train_generator, epochs=30, validation_data=valid_generator, 
	shuffle=False, max_queue_size=20, use_multiprocessing=True, workers=4)
print(model.evaluate_generator(evaluate_gen))
y_pred = model.predict_generator(predict_gen)
#print(y_pred[:], y_test)
y_pred = np.rint(y_pred)
y_test = np.array(y_test)
accuracy_s = sklearn.metrics.accuracy_score(y_test, y_pred)
auroc_s = sklearn.metrics.roc_auc_score(y_test, y_pred)
auprc_s = sklearn.metrics.average_precision_score(y_test, y_pred)
print(accuracy_s, auroc_s, auprc_s)

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('./figures/classifier.sequence.HepG2.accuracy.png')

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model binary entropy loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('./figures/classifier.sequence.HepG2.loss.png')
