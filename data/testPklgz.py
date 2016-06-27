import gzip
import pickle
with gzip.open('file_881.pkl.gz', 'rb') as f:
    train_set, valid_set, test_set = pickle.load(f)

train_set_x , train_set_y = train_set
val_set_x, val_set_y = valid_set

f = open("temp.txt","w")
for i in train_set_x[0]:
    f.write(str(i))
