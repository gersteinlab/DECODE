import numpy as np
import string
import random
import os
import pickle
import argparse

random.seed(0)

parser = argparse.ArgumentParser(description='enter the tracks and cell line for CNN')
parser.add_argument('-d', '--indir', type=str, help='working directory of the tab files')
parser.add_argument('-t', '--track', type=str, help='track of context information of length 100')
parser.add_argument('-c', '--cell', type=str, help='Name of the cell to investigate')
parser.add_argument('-r', '--ratio', type=int, default=10, help='positive and negative matched ratio')
args = parser.parse_args()

input_dir = args.indir
sig_track = args.track
cell_line = args.cell
pos_neg_ratio = args.ratio

pos_track_file = input_dir+"/"+cell_line+"."+sig_track+".positive.separated.tab"
neg_track_file = input_dir+"/"+cell_line+"."+sig_track+".negative.separated.tab"

print("reading in positive bigWigAverageOverBed tab output")
pos_track = np.genfromtxt(pos_track_file, dtype=str, delimiter='\t', usecols=5)#, max_rows=10000)

print("reading in negative bigWigAverageOverBed tab output")
neg_track = np.genfromtxt(neg_track_file, dtype=str, delimiter='\t', usecols=5)#, max_rows=10000 * pos_neg_ratio)

# print(pos_track[:10])
# print(neg_track[:10])

X_pos_train = list()
X_neg_train = list()
pos_train_length = pos_track.shape[0] // 100
pos_line_num = []
neg_line_num = []
for i in range(pos_train_length):
	random_number = i * 100
	pos_line_num.append(random_number)
	X_pos_train.append(pos_track[random_number:random_number+100].tolist())
	#pos_track = np.delete(pos_track, np.arange(random_number, random_number+100), 0)

	random_number_neg = random_number*pos_neg_ratio
	random_number_neg_idx = random_number_neg + (random.randint(0,pos_neg_ratio) * 100)
	neg_line_num.append(random_number_neg_idx)
	X_neg_train.append(neg_track[random_number_neg_idx:random_number_neg_idx+100].tolist())
	#neg_track = np.delete(neg_track, np.arange(random_number_neg, random_number_neg+(100*pos_neg_ratio)), 0)
	if i % 1000 == 0:
		print(i, random_number, random_number_neg)
		print(len(X_pos_train), len(X_neg_train))

np.savetxt("positive_line_number_"+sig_track+"_"+cell_line+".sorted.tsv", np.array(pos_line_num), delimiter='\t')
np.savetxt("negative_line_number_"+sig_track+"_"+cell_line+".sorted.tsv", np.array(neg_line_num), delimiter='\t')

print(X_pos_train[0])
print(X_neg_train[0])
print(pos_track.shape)
print(neg_track.shape)
print(len(X_pos_train))
print(len(X_neg_train))

print("combining positive and negative")

X_train = X_pos_train + X_neg_train
print(len(X_train))
y_train = [1] * pos_train_length + [0] * pos_train_length
print(len(y_train))

print("outputing results without shuffling")

f = open("encoded_"+sig_track+"_"+cell_line+".sorted.pickle", 'wb')
pickle.dump([X_train, y_train], f)
f.close()

