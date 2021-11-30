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

from sklearn.metrics import roc_auc_score

def auroc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc

def get_sequence(filename):
	print("reading pickle file: %s" % filename)
	f = open(filename, 'rb')
	total_seq_encoded, total_seq_label = pickle.load(f)
	f.close()

	print(total_seq_encoded[0][0], len(total_seq_encoded))
	print(total_seq_label[0], len(total_seq_label))

	print("partitioning training and test")
	X_train, X_test, y_train, y_test = train_test_split(
		total_seq_encoded, 
		total_seq_label, 
		test_size=0.1, random_state=42)

	print(len(X_train), len(X_train[0]), len(X_train[0][0]))

	return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = get_sequence("encoded_sequence.pickle")

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
        batch_x_flattened = np.array([val for sublist in batch_x for val in sublist])
        batch_x_flipped = batch_x_flattened.reshape(1,4000)

        batch_y = np.array(self.y[idx]).reshape(1,1)

        return batch_x_flipped, batch_y

train_generator = MY_Generator(X_train, y_train)
valid_generator = MY_Generator(X_test, y_test)

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
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', auroc])

model.fit_generator(train_generator, epochs=10, validation_data=valid_generator, shuffle=False, max_queue_size=20)








