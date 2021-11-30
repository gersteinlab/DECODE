import numpy as np
import string
import random
import os
import pickle

random.seed(0)

pos_neg_ratio = 10

pos_dnase_file = "/gpfs/ysm/scratch60/zc264/ChromVar/enhancer-prediction/encode/datasets/result_bed/HepG2.ChIP-seq.H3K27ac.positive.separated.tab"
neg_dnase_file = "/gpfs/ysm/scratch60/zc264/ChromVar/enhancer-prediction/encode/datasets/result_bed/HepG2.ChIP-seq.H3K27ac.negative.separated.tab"

print("reading in positive DNase")
pos_dnase = np.genfromtxt(pos_dnase_file, dtype=str, delimiter='\t', usecols=5)#, max_rows=10000)

print("reading in negative DNase")
neg_dnase = np.genfromtxt(neg_dnase_file, dtype=str, delimiter='\t', usecols=5)#, max_rows=10000 * pos_neg_ratio)

# print(pos_dnase[:10])
# print(neg_dnase[:10])

print(pos_dnase.shape)
print(neg_dnase.shape)

validation_size = 1000
X_pos_test = list()
X_neg_test = list()
for i in range(validation_size):
	peak_num = pos_dnase.shape[0] // 100

	random_number = random.randrange(int(peak_num)) * 100
	X_pos_test.append(pos_dnase[random_number:random_number+100].tolist())
	pos_dnase = np.delete(pos_dnase, np.arange(random_number, random_number+100), 0)

	random_number_neg = random_number*pos_neg_ratio
	X_neg_test.append(neg_dnase[random_number_neg:random_number_neg+100].tolist())
	neg_dnase = np.delete(neg_dnase, np.arange(random_number, random_number+100), 0)
	if i % 1000 == 0:
		print(i, random_number, random_number_neg)
		print(len(X_pos_test), len(X_neg_test))


print(X_pos_test[0])
print(X_neg_test[0])
print(pos_dnase.shape)
print(neg_dnase.shape)
print(len(X_pos_test), len(X_neg_test))


X_pos_train = list()
X_neg_train = list()
pos_train_length = pos_dnase.shape[0] // 100
for i in range(pos_train_length):
	peak_num = pos_dnase.shape[0] // 100
	
	random_number = random.randrange(int(peak_num)) * 100
	X_pos_train.append(pos_dnase[random_number:random_number+100].tolist())
	pos_dnase = np.delete(pos_dnase, np.arange(random_number, random_number+100), 0)

	random_number_neg = random_number*pos_neg_ratio
	X_neg_train.append(neg_dnase[random_number_neg:random_number_neg+100].tolist())
	neg_dnase = np.delete(neg_dnase, np.arange(random_number, random_number+100), 0)
	if i % 1000 == 0:
		print(i, random_number, random_number_neg)
		print(len(X_pos_train))
		print(len(X_neg_train))


print(X_pos_train[0])
print(X_neg_train[0])
print(pos_dnase.shape)
print(neg_dnase.shape)
print(len(X_pos_train))
print(len(X_neg_train))

print("combining positive and negative")

X_test = X_pos_test + X_neg_test
print(len(X_test))
y_test = [1] * validation_size + [0] * validation_size
print(len(y_test))


X_train = X_pos_train + X_neg_train
print(len(X_train))
y_train = [1] * pos_train_length + [0] * pos_train_length
print(len(y_train))

print("shuffling samples")
random.seed(0)
test_zipped = list(zip(X_test, y_test))
random.shuffle(test_zipped)
X_test, y_test = zip(*test_zipped)

train_zipped = list(zip(X_train, y_train))
random.shuffle(train_zipped)
X_train, y_train = zip(*train_zipped)

f = open("encoded_H3K27ac_HepG2.pickle", 'wb')
pickle.dump([X_train, y_train, X_test, y_test], f)
f.close()

