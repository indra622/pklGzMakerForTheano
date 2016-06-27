#-*- coding: utf-8 -*-

from PIL import Image
from numpy import genfromtxt
import gzip, cPickle
from glob import glob
import numpy as np
import pandas as pd

# def dir_to_dataset(glob_files, loc_train_labels=""):
def dir_to_dataset(glob_files):
    print("Gonna process:\n\t %s"%glob_files)
    dataset = []
    for file_count, file_name in enumerate( sorted(glob(glob_files),key=len) ):
        image = Image.open(file_name)
        img = Image.open(file_name).convert('LA') #tograyscale
        pixels = [f[0] for f in list(img.getdata())]
        dataset.append(pixels)
        if file_count % 1000 == 0:
            print("\t %s files processed"%file_count)

    return np.array(dataset)
    # outfile = glob_files+"out"
    # np.save(outfile, dataset)
    # if len(loc_train_labels) > 0:
    #     df = pd.read_csv(loc_train_labels)
    #     return np.array(dataset), np.array(df["Class"])
    # else:
    #     return np.array(dataset)

#start write pkl.gz file
f = gzip.open('file_out.pkl.gz','wb')

# insert input data and class

#1st class
Data = dir_to_dataset("class1/*.jpg")
#np.random.shuffle(Data) #random shuffle readed data
trn_size = 60000 #train data size
val_size = 10000 #validation data size
test_size = 30000 #test data size
val_idx = trn_size+1+val_size
test_idx = val_idx+1+test_size
data_size = trn_size+val_size+test_size #Total size of data

# Data and labels are created based on the specific number

#initialization list
train_set_y = list()
val_set_y = list()
test_set_y = list()

#create data matrix
train_set_x = Data[:trn_size]
val_set_x = Data[trn_size+1:val_idx]
test_set_x = Data[val_idx+1:data_size]

#give class num
for i in range(trn_size):
	train_set_y.append('0')
for i in range(val_size):
	val_set_y.append('0')
for i in range(test_idx):
	test_set_y.append('0')


#2nd class
Data = dir_to_dataset("class2/*.jpg")
# Data and labels are read  #random shuffle readed data
#np.random.shuffle(Data)
trn_size = 60000	#train data size
val_size = 10000	#validation data size
test_size = 30000	#test data size
val_idx = trn_size+1+val_size
test_idx = val_idx+1+test_size
data_size = trn_size+val_size+test_size #Total size of data

# Data and labels are created based on the specific number
#create data matrix using 1st class number
train_set_x = np.concatenate((train_set_x,Data[:trn_size]))
val_set_x = np.concatenate((val_set_x,Data[trn_size+1:val_idx]))
test_set_x = np.concatenate((test_set_x,Data[val_idx+1:data_size]))

#give class num
for i in range(trn_size):
    train_set_y.append('1')
for i in range(val_size):
    val_set_y.append('1')
for i in range(test_idx):
    test_set_y.append('1')


#shuffle train, val, test data, respectively
train_set_x2 = list()
train_set_y2 = list()
val_set_x2 = list()
val_set_y2 = list()
test_set_x2 = list()
test_set_y2 = list()

print len(train_set_x)
arr = np.arange(len(train_set_x))
np.random.shuffle(arr)

for i in arr:
    train_set_x2.append(train_set_x[i])
    train_set_y2.append(train_set_y[i])

print "train shuffling completed...."
arr = np.arange(len(val_set_x))
np.random.shuffle(arr)

for i in arr:
    val_set_x2.append(val_set_x[i])
    val_set_y2.append(val_set_y[i])

print "validation shuffling completed...."
arr = np.arange(len(test_set_x))
np.random.shuffle(arr)

for i in arr:
    test_set_x2.append(test_set_x[i])
    test_set_y2.append(test_set_y[i])

print "test shuffling completed....."

train_set = train_set_x2, train_set_y2
val_set = train_set_x2, train_set_y2
test_set = test_set_x2, test_set_y2

dataset = [train_set, val_set, test_set]
print "dataset merged"
print "dumping...."
cPickle.dump(dataset, f, protocol=2)
print "completed!"
f.close()
