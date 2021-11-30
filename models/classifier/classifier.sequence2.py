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

print("reading pickle file")
f = open("encoded_sequence2.pickle", 'rb')
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

class MY_Generator(Sequence):

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        x_len = len(self.X)
        #print(x_len)
        return x_len

    def __getitem__(self, idx):
        print(idx)

        batch_x = self.X[idx]
        print(len(batch_x))
        # batch_x = np.array([np.array(xi) for xi in batch_x])
        # print(batch_x.shape)
        # batch_x = batch_x.reshape((1, 1000, 4))
        # print(batch_x.shape)
        batch_x_flattened = np.array(batch_x)
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

# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit_generator(train_generator, epochs=10, validation_data=valid_generator)








