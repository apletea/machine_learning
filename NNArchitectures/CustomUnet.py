import cv2
import numpy as np
import random
import skimage
from skimage import data
from sklearn.model_selection import train_test_split
import pandas as pd
from matplotlib import pyplot as plt
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
from sklearn.metrics import roc_curve, auc, average_precision_score, precision_recall_curve
from sklearn.utils import class_weight
from keras.models import *
from keras.optimizers import *
from keras.applications import *

from keras.utils import plot_model

TRAIN_FOLDER = './train/'
TEST_FOLDER = './test/'
NAME_MODEL = 'Dense_201'





def double_conv_layer(x, size, dropout, batch_norm):
  if K.image_dim_ordering() == 'th':
    axis = 1
  else:
    axis = 3
  conv = Convolution2D(size, (3, 3), padding='same')(x)
  if batch_norm is True:
    conv = BatchNormalization(axis=axis)(conv)
  conv = Activation('relu')(conv)
  conv = Convolution2D(size, (3, 3), padding='same')(conv)
  if batch_norm is True:
    conv = BatchNormalization(axis=axis)(conv)
  conv = Activation('relu')(conv)
  if dropout > 0:
     conv = Dropout(dropout)(conv)
  return conv

def double_conv_conc(x, size, dropout, batch_norm):
  conv = BatchNormalization()(x)
  conv = Activation('relu')(conv)
  conv = Convolution2D(size, (3, 3), padding='same')(conv)  

def MiddleFlow(input, num_output):
  a = SeparableConv2D(num_output, 3, activation='relu', padding='same')(input)
  a = BatchNormalization()(a)
  a = SeparableConv2D(num_output, 3, activation='relu', padding='same')(a)
  a = BatchNormalization()(a)
  a = SeparableConv2D(num_output, 3, activation='relu', padding='same')(a)
  a = BatchNormalization()(a)
  out = Add()([a,input])
  return out

def EntryFlow(input, num_output):
  a = SeparableConv2D(num_output, 3, activation='relu', padding='same')(input)
  a = BatchNormalization()(a)
  a = SeparableConv2D(num_output, 3,activation='relu', padding='same')(a)
  a = BatchNormalization()(a)
  a = MaxPooling2D(2)(a)
  b = Convolution2D(num_output, 1, strides=2, activation='relu')(input)
  b = BatchNormalization()(b)
  out = Add()([a,b])
  return out

def get_model(input_shape=(112,112,3)):
  filters = 32
  dropout = False
  batch_norm = True
  img_input = Input(shape=input_shape)

  conv112 = double_conv_layer(img_input, filters, dropout, batch_norm)

  pool56 = EntryFlow(conv112, filters*2)
  conv56 = MiddleFlow(pool56, filters*2)
  conv56 = MiddleFlow(conv56, filters*2)
  conv56 = MiddleFlow(conv56, filters*2)  

  pool28 = EntryFlow(conv56,  filters*4)
  conv28 = MiddleFlow(pool28, filters*4)
  conv28 = MiddleFlow(conv28, filters*4) 
  conv28 = MiddleFlow(conv28, filters*4) 

  pool14 = EntryFlow(conv28,  filters*8)
  conv14 = MiddleFlow(pool14, filters*8)
  conv14 = MiddleFlow(conv14, filters*8) 
  conv14 = MiddleFlow(conv14, filters*8) 

  pool7 = EntryFlow(conv14,   filters*16)
  conv7 = MiddleFlow(pool7, filters*16) 
  conv7 = MiddleFlow(conv7, filters*16) 
  conv7 = MiddleFlow(conv7, filters*16)

  up14 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv14], axis=-1)
  up14 = Convolution2D(filters*8, (1, 1), activation='relu')(up14)
  conv14 = MiddleFlow(up14, filters*8)
  conv14 = MiddleFlow(conv14, filters*8) 
  conv14 = MiddleFlow(conv14, filters*8)

  up28 = concatenate([UpSampling2D(size=(2, 2))(conv14), conv28], axis=-1)
  up28 = Convolution2D(filters*4, (1, 1), activation='relu')(up28)
  conv28 = MiddleFlow(up28, filters*4)
  conv28 = MiddleFlow(conv28, filters*4) 
  conv28 = MiddleFlow(conv28, filters*4) 

  up56 = concatenate([UpSampling2D(size=(2, 2))(conv28), conv56], axis=-1)
  up56 = Convolution2D(filters*2, (1, 1), activation='relu')(up56)
  conv56 = MiddleFlow(up56, filters*2)
  conv56 = MiddleFlow(conv56, filters*2)
  conv56 = MiddleFlow(conv56, filters*2) 

  up112 = concatenate([UpSampling2D(size=(2, 2))(conv56), conv112], axis=-1)
  conv112 = Convolution2D(1, (1, 1), activation='sigmoid')(up112)
  
  model = Model(img_input,conv112)
  model.compile(loss='binary_crossentropy', optimizer='sgd')
  return model

