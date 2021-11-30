import numpy as np
import string
import random
import os
from sklearn.model_selection import train_test_split
import pickle
import argparse
import math
from datetime import datetime
import sklearn

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
from keras.callbacks import EarlyStopping, Callback, TensorBoard
from matplotlib import pyplot as plt
import keras_metrics as km


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

	# get data per track

    print("reading pickle file: %s" % filename)
    f = open(filename, 'rb')
    X_train, y_train = pickle.load(f)
    f.close()

    return X_train, y_train

def get_data(tracks_list, cell, in_order=False):

	# get for all the tracks

	track_X_list = list() #[H3K27ac (x, 100), DNase (x, 100), ...] 
	for track in tracks_list:
		if in_order == True:
			filename=args.input_dir+"/encoded_" + track + "_" + cell + ".sorted.pickle"
		else:
			filename=args.input_dir+"/encoded_" + track + "_" + cell + ".pickle"
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

	# load the model

	K.clear_session()
	pool2_list = []
	merge_list = []
	input_list = []
	for track in tracks_list:
		input_size = Input(shape=(1, 100, 1))
		input_list.append(input_size)
		conv1_ = Activation(activation='relu')(Conv2D(128, (1, 10), padding='same')(input_size))
		conv1  = SqueezeExcite(conv1_)
		merge_list.append(conv1)

		conv2_ = Activation(activation='relu')(Conv2D(64, (1, 1), padding='same')(conv1))
		conv2  = SqueezeExcite(conv2_)
		conv3_ = Activation(activation='relu')(Conv2D(64, (1, 3), padding='same')(conv2))
		conv3  = SqueezeExcite(conv3_)
		conv4_ = Activation(activation='relu')(Conv2D(128, (1, 1), padding='same')(conv3))
		conv4  = SqueezeExcite(conv4_)
		pool1  = MaxPooling2D(pool_size=(1, 2))(conv4)
		conv5_ = Activation(activation='relu')(Conv2D(64, (1, 3), padding='same')(pool1))
		conv5  = SqueezeExcite(conv5_)
		conv6_ = Activation(activation='relu')(Conv2D(64, (1, 3), padding='same')(conv5))
		conv6  = SqueezeExcite(conv6_)
		conv7_ = Activation(activation='relu')(Conv2D(128, (1, 1), padding='same')(conv6))
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

def all_data(input_dir, cell, in_order=False):

	# obtain all of the data in the pickle files

	X = list() #[H3K27ac (x, 100), DNase (x, 100), ...] 
	for track in tracks_list:
		if in_order == True:
			filename=args.input_dir+"/encoded_" + track + "_" + cell + ".sorted.pickle"
		else:
			filename=args.input_dir+"/encoded_" + track + "_" + cell + ".pickle"
		print("reading pickle file: %s" % filename)
		f = open(filename, 'rb')
		X_train, y = pickle.load(f)
		f.close()
		X.append(X_train)

	x_train = list()
	for x in X:
		x_train.append(np.array([np.array(x[i]) for i in range(len(y))]).reshape((-1, 1, 100, 1)))
	y_train = np.array([np.array(y[i]) for i in range(len(y))]).reshape(-1,1)

	print(x_train[0].shape)
	print(y_train.shape)

	return x_train, y_train


if __name__ == '__main__':

	#parsing command line arguments
	parser = argparse.ArgumentParser(description='enter the tracks and cell line for CNN')
	parser.add_argument('-t', '--tracks', type=str, help='track of context information of length 100')
	parser.add_argument('-a', '--cellA', type=str, help='Name of the A cell with trained model')
	parser.add_argument('-b', '--cellB', type=str, help='Name of the B cell to cross evaluate')
	parser.add_argument('-i', '--input_dir', type=str, help='pickle file input directory')
	parser.add_argument('-n', '--name', type=str, help='output file root name')
	args = parser.parse_args()
	tracks_list = args.tracks.split(',')

	#structing input output name stems
	track_name = '+'.join(tracks_list)
	A_name = 'classifier.'+args.name+"."+track_name+'.'+args.cellA
	B_name = 'classifier.'+args.name+"."+track_name+'.'+args.cellB
	output_name = 'classifier.se.'+args.name+"."+track_name+'.'+args.cellA + '.' + args.cellB
	figure_output_name = './figures/'+output_name

	#obtain all of the data (no kfold division)
	x_train, y_train = all_data(args.input_dir, args.cellA)
	x_test, y_test = all_data(args.input_dir, args.cellB, True)

	#get the model going
	model = load_model(tracks_list)
	adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=8e-4)
	model.compile(loss='binary_crossentropy', optimizer=adam, 
		metrics=['accuracy', auroc, auprc, f1_m, recall_m, precision_m])

	#train the model
	history = model.fit(x_train, y_train,
				batch_size=1,
				epochs=30,
				validation_split=0.1)

	#save the model and the weights in case if the model doesn't work
	os.system("mkdir -p ./model/"+output_name)
	model.save_weights('./model/'+output_name+'/' + output_name + '.weights.h5')
	model.save('./model/'+output_name+'/' + output_name + '.h5')

    #predict the enhancers on cell B
	y_pred = model.predict(x_test)
	print(y_pred.shape)
	#model.evaluate(x_train, y_train)
	y_out = y_pred.reshape(-1)
	print(y_out.shape)
	np.savetxt('./pred_result/'+output_name+'.tsv', y_out, delimiter='\n')

	#measuring the accuracy enhancer prediction in cell B
	accuracy_s = sklearn.metrics.accuracy_score(y_test, np.rint(y_pred))

	# plot accuracy over time in cell A
	plt.figure()
	plt.plot(history.history['acc'])
	plt.plot(history.history['val_acc'])
	plt.title('model accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train', 'val'], loc='upper left')
	plt.savefig(figure_output_name+'.accuracy.png')

	# plot loss over time in cell A
	plt.figure()
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('model binary entropy loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'val'], loc='upper left')
	plt.savefig(figure_output_name+'.loss.png')

	# auroc over time in cell A
	plt.figure()
	plt.plot(history.history['auroc'])
	plt.plot(history.history['val_auroc'])
	plt.title('model auROC')
	plt.ylabel('auroc')
	plt.xlabel('epoch')
	plt.legend(['train', 'val'], loc='upper left')
	plt.savefig(figure_output_name+'.auROC.png')

	# auprc over time in cell A
	plt.figure()
	plt.plot(history.history['auprc'])
	plt.plot(history.history['val_auprc'])
	plt.title('model auPRC')
	plt.ylabel('auprc')
	plt.xlabel('epoch')
	plt.legend(['train', 'val'], loc='upper left')
	plt.savefig(figure_output_name+'.auPRC.png')

	# ROC in test set (cell B)
	plt.figure()
	fpr_keras, tpr_keras, thresholds_keras = sklearn.metrics.roc_curve(y_test, y_pred)
	auroc_s = sklearn.metrics.auc(fpr_keras, tpr_keras)
	plt.plot([0, 1], [0, 1], 'k--')
	plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auroc_s))
	plt.xlabel('False positive rate')
	plt.ylabel('True positive rate')
	plt.title('ROC curve')
	plt.legend(loc='best')
	plt.savefig(figure_output_name+'.ROC.png')

	# PRC in test set (cell B)
	plt.figure()
	precision_keras, recall_keras, thresholds_keras = sklearn.metrics.precision_recall_curve(y_test, y_pred)
	auprc_s = sklearn.metrics.auc(recall_keras, precision_keras)
	plt.plot([0, 1], [0, 1], 'k--')
	plt.plot(recall_keras, precision_keras, label='Keras (area = {:.3f})'.format(auprc_s))
	plt.xlabel('Precision')
	plt.ylabel('Recall')
	plt.title('PR curve')
	plt.legend(loc='best')
	plt.savefig(figure_output_name+'.PR.png')

	print(accuracy_s, auroc_s, auprc_s)

