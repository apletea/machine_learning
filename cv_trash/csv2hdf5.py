
import csv
import h5py
import numpy as np


def encode_pixel(pixel_value):
    return float(pixel_value) / 256.0


def encode_direction(direction):
    return {"up": 0,
            "straight": 1,
            "left": 2,
            "right": 3,
            }[direction]


def data_matrix_from_csv(csv_filename,i,step):
    data = pd.read_csv(csv_filename,names=['x','y','img'],skiprows=i+1,nrows=step,header=None)
    return data


def data_and_labels(data_matrix):
    print data_matrix
    #np.random.shuffle(data_matrix)
    data = data_matrix['img']
    labels = data_matrix[['x','y']]
    return data, labels


def save_dataset_with(filename, data, labels, i):
    print data.astype("f8")
    f = h5py.File(filename, "a")
    if (i == 0):
        f.create_dataset("data", data.shape, dtype="f8")
        f.create_dataset("label", labels.shape, dtype="i4")
   # f["data"][:] = data.astype("f8")
    #f["label"][:] = labels.astype("i4")
    f.close

for i in xrange(0,1384,10):
    print i
    data, labels = data_and_labels(data_matrix_from_csv("res.csv",i,10))
    #print data
    #print labels
    save_dataset_with("faces-train.h5", data[0:8], labels[0:8],i)
    save_dataset_with("faces-test.h5", data[8:], labels[8:],i)
