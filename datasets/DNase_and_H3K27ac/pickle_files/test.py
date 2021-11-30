import numpy
import pickle

infile=open("encoded_DNase_K562.sorted.pickle","rb")
a=pickle.load(infile)
infile.close()

for x, y in enumerate(a[0]):
	if len(y) == 0:
		print(x)
