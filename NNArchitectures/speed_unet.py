import cv2
import numpy as np
import random
import skimage
#from skimage import data
#from sklearn.model_selection import train_test_split
import pandas as pd
#from matplotlib import pyplot as plt
import os
import glob
import tensorflow as tf
import keras

config = tf.ConfigProto( device_count = {'GPU' : 1, 'CPU':8})
sess = tf.Session(config=config)
keras.backend.set_session(sess)

from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras.applications import *
from keras.regularizers import *
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.models import load_model
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint , EarlyStopping, ReduceLROnPlateau, TensorBoard
from keras.utils.np_utils import to_categorical
from keras import backend as K
from keras.models import load_model
#from sklearn.metrics import roc_curve, auc, average_precision_score, precision_recall_curve
#from sklearn.utils import class_weight
from keras.models import *
from keras.optimizers import *
from keras.applications import *

#from keras.utils import plot_model

TRAIN_FOLDER = './train/'
TEST_FOLDER = './test/'
NAME_MODEL = 'Dense_201'





def double_conv_layer(x, size, dropout, batch_norm):

  conv = Convolution2D(size, (7, 7), strides=(4,4), padding='same')(x)
  conv = Activation('relu')(conv)
  conv = BatchNormalization()(x)


  return conv

def double_conv_conc(x, size, dropout, batch_norm):
  conv = BatchNormalization()(x)
  conv = Activation('relu')(conv)
  conv = Convolution2D(size, (3, 3), padding='same')(conv)  

def MiddleFlow(input, num_output):
 # a = Convolution2D(num_output/4, 1, activation='relu', padding='same')(input)
  a = Convolution2D(num_output/32, 1, activation='relu', padding='same')(input)
#  a = Convolution2D(num_output/32, (3,1), activation='relu', padding='same')(a)
  a = Convolution2D(num_output/32, 3, activation='relu', padding='same')(a)
  a = Convolution2D(num_output, 1, activation='relu', padding='same')(a)
  a = BatchNormalization()(a)
  out = Add()([a,input])
  return out

def EntryFlow(input, num_output, kernel = 4, stride=(4,4)):
  a = Convolution2D(num_output/32, 1, activation='relu', padding='same')(input)
  a = Convolution2D(num_output/32, kernel, strides=stride, activation='relu')(a)
  a = Convolution2D(num_output, 1, activation='relu', padding='same')(a)
  
  b = Convolution2D(num_output, 1, activation='relu', padding='same')(input)
  b = MaxPooling2D(kernel, strides=stride)(b)
  out = Add()([a,b])
  out = BatchNormalization()(out)
  return out

def с(input_shape=(299,299,3), num_classes=5):
  filters = 32 
  dropout = False
  batch_norm = True
  img_input = Input(shape=input_shape)

#
  conv112 = img_input
  
  #299 -> 74
  conv56 = Convolution2D(filters,7, strides=(4,4),activation='relu', padding='valid')(conv112)

  #74 -> 12
  pool28 = EntryFlow(conv56,  filters*2, 8, (6,6))
  conv28 = MiddleFlow(pool28, filters*2) 
  conv28 = MiddleFlow(conv28, filters*2)  
  conv28 = MiddleFlow(conv28, filters*2)
  
  #12 -> 4
  pool14 = EntryFlow(conv28,  filters*4,3,3)
  conv14 = MiddleFlow(pool14, filters*4)
  conv14 = MiddleFlow(conv14, filters*4)
  conv14 = MiddleFlow(conv14, filters*4)

  #18 -> 4
 # pool7 = EntryFlow(conv14,   filters*8)
 # conv7 = MiddleFlow(pool7, filters*8) 


 # up14 = concatenate([ZeroPadding2D(1)(UpSampling2D(size=(4, 4))(conv7)), conv14], axis=-1)
  #up14 = Convolution2D(filters*4, (1, 1), activation='relu')(up14)
  #4 -> 12
  up28 = concatenate([(UpSampling2D(size=(3, 3))(conv14)), conv28], axis=-1)
  up28 = Convolution2D(filters*2, (1, 1), activation='relu')(up28)
  up28 = MiddleFlow(up28, filters*2) 

  #12 -> 73
  up56 = concatenate([ZeroPadding2D((1))(UpSampling2D(size=(6, 6))(up28)), conv56], axis=-1)
  up56 = Convolution2D(filters, (1, 1), activation='relu')(up56)
  up56 = MiddleFlow(up56, filters) 
  #73 -> 299
  up112 = concatenate([ZeroPadding2D(((2,1),(2,1)))(UpSampling2D(size=(4, 4))(up56)),(conv112)], axis=-1)
  up112 = Convolution2D(num_classes, (1, 1), activation='relu')(up112)
  
  model = Model(img_input,up112)
  model.compile(loss='binary_crossentropy', optimizer='sgd')
  return model

import time
import numpy as np
model = get_model()
print model.summary()

zeros = np.zeros((1,299,299,3))
start = time.time()
for i in range(100):
  model.predict(zeros)
end = time.time()
print(((end - start)/100.0))
