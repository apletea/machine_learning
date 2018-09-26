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
  a = Convolution2D(num_output/16, 1, activation='relu', padding='same')(input)
  a = Convolution2D(num_output/16, 3, activation='relu', padding='same')(a)
  a = Convolution2D(num_output, 1, activation='relu', padding='same')(a)
  a = BatchNormalization()(a)
  out = Add()([a,input])
  return out

def EntryFlow(input, num_output):
  a = Convolution2D(num_output/16, 1, activation='relu', padding='same')(input)
  a = Convolution2D(num_output/16, 6, strides=(4,4), activation='relu')(a)
  a = Convolution2D(num_output, 1, activation='relu', padding='same')(a)
  
  b = Convolution2D(num_output, 1, activation='relu', padding='same')(input)
  b = MaxPooling2D(6, strides=(4,4))(b)
  out = Add()([a,b])
  out = BatchNormalization()(out)
  return out

def get_model(input_shape=(1198,1198,3)):
  filters = 32
  dropout = False
  batch_norm = True
  img_input = Input(shape=input_shape)

#
  conv112 = Convolution2D(filters*2,1,activation='relu', padding='same')(img_input)
  
  #1198 -> 299
  pool56 = EntryFlow(conv112, filters*2)
  conv56 = MiddleFlow(pool56, filters*2)
  conv56 = MiddleFlow(conv56, filters*2)
  conv56 = MiddleFlow(conv56, filters*2)  

  #299 -> 74
  pool28 = EntryFlow(conv56,  filters*4)
  conv28 = MiddleFlow(pool28, filters*4)
  conv28 = MiddleFlow(conv28, filters*4) 
  conv28 = MiddleFlow(conv28, filters*4) 
  
  #74 -> 18
  pool14 = EntryFlow(conv28,  filters*8)
  conv14 = MiddleFlow(pool14, filters*8)
  conv14 = MiddleFlow(conv14, filters*8) 
  conv14 = MiddleFlow(conv14, filters*8) 
  
  #18 -> 4
  pool7 = EntryFlow(conv14,   filters*16)
  conv7 = MiddleFlow(pool7, filters*16) 
  conv7 = MiddleFlow(conv7, filters*16) 
  conv7 = MiddleFlow(conv7, filters*16)

  up14 = concatenate([ZeroPadding2D(1)(UpSampling2D(size=(4, 4))(conv7)), conv14], axis=-1)
  up14 = Convolution2D(filters*8, (1, 1), activation='relu')(up14)
  conv14 = MiddleFlow(up14, filters*8)


  up28 = concatenate([ZeroPadding2D(1)(UpSampling2D(size=(4, 4))(conv14)), conv28], axis=-1)
  up28 = Convolution2D(filters*4, (1, 1), activation='relu')(up28)
  conv28 = MiddleFlow(up28, filters*4)


  up56 = concatenate([ZeroPadding2D(((1,2),(1,2)))(UpSampling2D(size=(4, 4))(conv28)), conv56], axis=-1)
  up56 = Convolution2D(filters*2, (1, 1), activation='relu')(up56)
  conv56 = MiddleFlow(up56, filters*2)


  up112 = concatenate([ZeroPadding2D(1)(UpSampling2D(size=(4, 4))(conv56)), conv112], axis=-1)
  conv112 = Convolution2D(1, (1, 1), activation='sigmoid')(up112)
  
  model = Model(img_input,conv112)
  model.compile(loss='binary_crossentropy', optimizer='sgd')
  return model

import time
import numpy as np
model = get_model()
print model.summary()

zeros = np.zeros((1,1198,1198,3))
start = time.time()
for i in range(100):
  model.predict(zeros)
end = time.time()
print(((end - start)/100.0))
