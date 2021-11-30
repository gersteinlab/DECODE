import numpy as np
import string
import random
import os
import pickle

random.seed(0)


print("reading in sequences")

pos_seq_file = "/gpfs/ysm/scratch60/zc264/ChromVar/enhancer-prediction/encode/datasets/matched/sorted.positive.HepG2.matched.data.tsv"
neg_seq_file = "/gpfs/ysm/scratch60/zc264/ChromVar/enhancer-prediction/encode/datasets/matched/sorted.negative.HepG2.matched.data.tsv"

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
	pos_seq_encoded_list = [return_encode(p) for p in pos_seq_split[i]]
	pos_seq_encoded.append(pos_seq_encoded_list)
	if i % 1000 == 0:
		print(i)
		#print(len(pos_seq_encoded_list))
	

# print(pos_seq_encoded[0].shape)
# print(pos_seq_split[0][0:10])
# print(pos_seq_encoded[0][0:10])

neg_seq_encoded = list()
print("total length: %d" % len(neg_seq_split))
for i in range(len(neg_seq_split)):
	neg_seq_encoded_list = [return_encode(p) for p in neg_seq_split[i]]
	neg_seq_encoded.append(neg_seq_encoded_list)
	if i % 1000 == 0:
		print(i)

validation_size = 1000
print(len(pos_seq_encoded))
print(len(neg_seq_encoded))


pos_neg_ratio = 10
X_pos_test = list()
X_neg_test = list()
for i in range(validation_size):
	random_number = random.randrange(len(pos_seq_encoded))
	X_pos_test.append(pos_seq_encoded.pop(random_number))
	X_neg_test.append(neg_seq_encoded.pop(random_number * pos_neg_ratio))
print(len(pos_seq_encoded))
print(len(neg_seq_encoded))


print(len(pos_seq_encoded))
print(len(neg_seq_encoded))
random.seed(0)
X_pos_train = list()
X_neg_train = list()
pos_train_length = len(pos_seq_encoded)
for i in range(pos_train_length):
	random_number = random.randrange(len(pos_seq_encoded))
	X_pos_train.append(pos_seq_encoded.pop(random_number))
	X_neg_train.append(neg_seq_encoded.pop(random_number * pos_neg_ratio))
print(len(pos_seq_encoded))
print(len(neg_seq_encoded))

print("combining positive and negative")

X_test = X_pos_test + X_neg_test
print(len(X_test))
y_test = [1] * validation_size + [0] * validation_size
print(len(y_test))

X_train = X_pos_train + X_neg_train
print(len(X_train))
y_train = [1] * pos_train_length + [0] * pos_train_length
print(len(y_train))

# X_train = pos_seq_encoded + neg_seq_encoded
# print(len(X_train))
# y_train = [1] * len(pos_seq_encoded) + [0] * len(neg_seq_encoded)
# print(len(y_train))


print("shuffling samples")
random.seed(0)
test_zipped = list(zip(X_test, y_test))
random.shuffle(test_zipped)
X_test, y_test = zip(*test_zipped)

train_zipped = list(zip(X_train, y_train))
random.shuffle(train_zipped)
X_train, y_train = zip(*train_zipped)

f = open("encoded_sequence_HepG2.pickle", 'wb')
pickle.dump([X_train, y_train, X_test, y_test], f)
f.close()





