#-*- coding: utf-8 -*-

from PIL import Image
from numpy import genfromtxt
import gzip, cPickle
from glob import glob
import numpy as np
import pandas as pd

def dir_to_dataset(glob_files, loc_train_labels=""):
    print("Gonna process:\n\t %s"%glob_files)
    dataset = []
    for file_count, file_name in enumerate( sorted(glob(glob_files),key=len) ):
        image = Image.open(file_name)
        img = Image.open(file_name).convert('LA') #tograyscale
        pixels = [f[0] for f in list(img.getdata())]
        dataset.append(pixels)
        if file_count % 1000 == 0:
            print("\t %s files processed"%file_count)
    # outfile = glob_files+"out"
    # np.save(outfile, dataset)
    if len(loc_train_labels) > 0:
        df = pd.read_csv(loc_train_labels)
        return np.array(dataset), np.array(df["Class"])
    else:
        return np.array(dataset)


# 
Data, y = dir_to_dataset("trainMNISTForm\\*.BMP","trainLabels.csv")
# Data and labels are read 

train_set_x = Data[:2093]
val_set_x = Data[2094:4187]
test_set_x = Data[4188:6281]
train_set_y = y[:2093]
val_set_y = y[2094:4187]
test_set_y = y[4188:6281]
# Divided dataset into 3 parts. I had 6281 images.

Data, y = dir_to_dataset("trainMNISTForm\\*.BMP","trainLabels.csv")
# Data and labels are read 

train_set_x = Data[:2093]
val_set_x = Data[2094:4187]
test_set_x = Data[4188:6281]
train_set_y = y[:2093]
val_set_y = y[2094:4187]
test_set_y = y[4188:6281]
# Divided dataset into 3 parts. I had 6281 images.
train_set = train_set_x, train_set_y
val_set = val_set_x, val_set_y
test_set = test_set_x, val_set_y

dataset = [train_set, val_set, test_set]

f = gzip.open('file.pkl.gz','wb')
cPickle.dump(dataset, f, protocol=2)
f.close()