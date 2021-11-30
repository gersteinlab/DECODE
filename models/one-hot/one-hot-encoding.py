import numpy as np
import string
import random
import os
import pickle


print("reading in sequences")

pos_seq_file = "/gpfs/ysm/scratch60/zc264/ChromVar/enhancer-prediction/encode/datasets/matched/positive.matched.data.tsv"
neg_seq_file = "/gpfs/ysm/scratch60/zc264/ChromVar/enhancer-prediction/encode/datasets/matched/negative.matched.data.tsv"

pos_seq = np.genfromtxt(pos_seq_file, dtype=str, delimiter='\t', usecols=5)
neg_seq = np.genfromtxt(neg_seq_file, dtype=str, delimiter='\t', usecols=5)

# print(pos_seq[:10])
# print(neg_seq[:10])

print("encoding sequences")

pos_seq_split = map(str.upper, pos_seq)
pos_seq_split = list(map(list, pos_seq_split))

neg_seq_split = map(str.upper, neg_seq)
neg_seq_split = list(map(list, neg_seq_split))

#encode_dict = {'A': np.array([1, 0, 0, 0]), 'C': np.array([0, 1, 0, 0]), 'G': np.array([0, 0, 1, 0]), 'T': np.array([0, 0, 0, 1]), 'N': np.array([1, 1, 1, 1])}
encode_dict = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1], 'N': [1, 1, 1, 1]}
def return_encode(letter):
	if letter in encode_dict.keys():
		return encode_dict[letter]
	else:
		return [0, 0, 0, 0]

pos_seq_encoded = list()
print("total length: %d" % len(pos_seq_split))
for i in range(len(pos_seq_split)):
	if i % 1000 == 0:
		print(i)
	pos_seq_encoded.append([return_encode(p) for p in pos_seq_split[i]])

# print(pos_seq_encoded[0].shape)
# print(pos_seq_split[0][0:10])
# print(pos_seq_encoded[0][0:10])

neg_seq_encoded = list()
print("total length: %d" % len(neg_seq_split))
for i in range(len(neg_seq_split)):
	if i % 1000 == 0:
		print(i)
	neg_seq_encoded.append([return_encode(p) for p in neg_seq_split[i]])

total_seq_encoded = pos_seq_encoded + neg_seq_encoded
total_seq_label = [1] * len(pos_seq_encoded) + [0] * len(neg_seq_encoded)

f = open("encoded_sequence.pickle", 'wb')
pickle.dump([total_seq_encoded, total_seq_label], f)
f.close()







