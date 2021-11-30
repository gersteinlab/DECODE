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
args = parser.parse_args()

input_dir = args.indir
sig_track = args.track
cell_line = args.cell

pos_track_file = input_dir+"/"+cell_line+"."+sig_track+".separated.tab"

print("reading in positive bigWigAverageOverBed tab output")
pos_track = np.genfromtxt(pos_track_file, dtype=str, delimiter='\t', usecols=5)#, max_rows=10000)
#pos_track = pos_track - np.mean(pos_track) / np.std(pos_track)

# print(pos_track[:10])

X_pos_train = list()
pos_train_length = pos_track.shape[0] // 100
pos_line_num = []
for i in range(pos_train_length):
	idx_number = i * 100
	pos_line_num.append(idx_number)
	X_pos_train.append(pos_track[idx_number:idx_number+100].tolist())
	if i % 1000 == 0:
		print(i, idx_number)
		print(len(X_pos_train))


print(X_pos_train[0])
print(pos_track.shape)
print(len(X_pos_train))

print("outputing results without shuffling")
f = open(input_dir+"/"+"encoded_"+sig_track+"_"+cell_line+".sorted.pickle", 'wb')
pickle.dump([X_pos_train], f)
f.close()

