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

f = gzip.open('file_622.pkl.gz','wb')
# 클래스별 데이터 삽입
Data = dir_to_dataset("bike_resize/*.jpg")
np.random.shuffle(Data)
trn_size = 588
val_size = 168
val_idx = trn_size+1+val_size
data_size = 840
# Data and labels are read 

train_set_y = list()
val_set_y = list()
test_set_y = list()

train_set_x = Data[:trn_size]
val_set_x = Data[trn_size+1:val_idx]
test_set_x = Data[val_idx+1:data_size]
for i in range(trn_size):
	train_set_y.append('0')
for i in range(val_size):
	val_set_y.append('0')
	test_set_y.append('0')


Data = dir_to_dataset("bus_resize/*.jpg")
# Data and labels are read 
np.random.shuffle(Data)
trn_size = 1638
val_size = 546
val_idx = trn_size+1+val_size
data_size = 2730
# Data and labels are read 

train_set_x = np.concatenate((train_set_x,Data[:trn_size]))
val_set_x = np.concatenate((val_set_x,Data[trn_size+1:val_idx]))
test_set_x = np.concatenate((test_set_x,Data[val_idx+1:data_size]))
for i in range(trn_size):
    train_set_y.append('1')
for i in range(val_size):
    val_set_y.append('1')
    test_set_y.append('1')

Data = dir_to_dataset("sedan_resize/*.jpg")
# Data and labels are read 
np.random.shuffle(Data)

trn_size = 15498
val_size = 5166
val_idx = trn_size+1+val_size
data_size = 25830
# Data and labels are read 


train_set_x = np.concatenate((train_set_x,Data[:trn_size]))
val_set_x = np.concatenate((val_set_x,Data[trn_size+1:val_idx]))
test_set_x = np.concatenate((test_set_x,Data[val_idx+1:data_size]))
for i in range(trn_size):
    train_set_y.append('2')
for i in range(val_size):
    val_set_y.append('2')
    test_set_y.append('2')

Data = dir_to_dataset("truck_resize/*.jpg")
# Data and labels are read 
np.random.shuffle(Data)

trn_size = 7308
val_size = 2436
val_idx = trn_size+1+val_size
data_size = 12180
# Data and labels are read 

train_set_x = np.concatenate((train_set_x,Data[:trn_size]))
val_set_x = np.concatenate((val_set_x,Data[trn_size+1:val_idx]))
test_set_x = np.concatenate((test_set_x,Data[val_idx+1:data_size]))
for i in range(trn_size):
    train_set_y.append('3')
for i in range(val_size):
    val_set_y.append('3')
    test_set_y.append('3')

Data = dir_to_dataset("van_resize/*.jpg")
# Data and labels are read 
np.random.shuffle(Data)


trn_size = 13860
val_size = 4620
val_idx = trn_size+1+val_size
data_size = 23100
# Data and labels are read 

train_set_x = np.concatenate((train_set_x,Data[:trn_size]))
val_set_x = np.concatenate((val_set_x,Data[trn_size+1:val_idx]))
test_set_x = np.concatenate((test_set_x,Data[val_idx+1:data_size]))
for i in range(trn_size):
    train_set_y.append('4')
for i in range(val_size):
    val_set_y.append('4')
    test_set_y.append('4')

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
