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
from keras.optimizers import Adam
from keras.models import Sequential
from keras.utils import Sequence
from keras.layers import Dense

import sklearn

def auroc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc

def get_sequence(filename):
    print("reading pickle file: %s" % filename)
    f = open(filename, 'rb')
    X_train, y_train, X_test, y_test = pickle.load(f)
    f.close()

    return X_train, y_train, X_test, y_test

X_train, y_train, X_test, y_test = get_sequence("encoded_sequence_K562.pickle")
print(len(X_train), len(X_train[0]))
print(y_train[:100])
print(len(X_test), len(X_test[0]))
print(y_test[:])

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

        batch_x = np.array(self.X[idx])
        batch_x_flipped = batch_x.reshape(1,4000)

        batch_y = np.array(self.y[idx]).reshape(1,1)

        return batch_x_flipped, batch_y

train_generator = MY_Generator(X_train, y_train)
valid_generator = MY_Generator(X_test, y_test)
predict_gen = MY_Generator(X_test, y_test)
evaluate_gen = MY_Generator(X_test, y_test)

#define the keras model
model = Sequential()
model.add(Dense(2000, input_dim=4000, activation='relu'))
model.add(Dense(400, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(25, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# model.add(Dense(2000, input_dim=4000, activation='relu'))
# model.add(Dense(1000, activation='relu'))
# model.add(Dense(200, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))

# compile the keras model
adam = Adam(lr=1e-6, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss='binary_crossentropy', optimizer=adam, 
    metrics=['accuracy', auroc, 'binary_accuracy', 'sparse_categorical_accuracy'])

model.fit_generator(train_generator, epochs=1, validation_data=valid_generator, 
    shuffle=True, max_queue_size=20, steps_per_epoch=5000)
print(model.evaluate_generator(evaluate_gen))
y_pred = model.predict_generator(predict_gen)
print(y_pred)
# accuracy_score = sklearn.metrics.accuracy_score(y_test, y_pred)
# auroc_score = sklearn.metrics.roc_auc_score(y_test, y_pred)
# auprc_score = sklearn.metrics.average_precision_score(y_test, y_pred)
# print(accuracy_score, auroc_score, auprc_score)





