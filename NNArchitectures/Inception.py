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

def InceptionBlock_1(input, num_outputs):
  ince_1x1 = Convolution2D(num_outputs[0], 1, 1, border_mode='same',activation='relu')(input)
  ince_3x3_red = Convolution2D(num_outputs[1], 1, 1, border_mode='same',activation='relu')(input)
  ince_3x3 = Convolution2D(num_outputs[2], 3, 3, border_mode='same', activation='relu')(ince_3x3_red)
  ince_5x5_red = Convolution2D(num_outputs[3], 1, 1, border_mode='same', activation='relu')(input)
  ince_5x5 = Convolution2D(num_outputs[4], 3, 3, border_mode='same', activation='relu')(ince_5x5_red)
  ince_5x5 = Convolution2D(num_outputs[4], 3, 3, border_mode='same', activation='relu')(ince_5x5_red)
  ince_pool = MaxPooling2D(3,1,border_mode='same')(input)
  ince_pool_proj = Convolution2D(num_outputs[5], 1, 1, border_mode='same', activation='relu')(ince_pool)
  
  ince_out = merge([ince_1x1, ince_3x3, ince_5x5, ince_pool_proj], mode='concat', concat_axis=-1)
  return ince_out

def InceptionBlock_2(input, num_outputs):
  ince_1x1 = Convolution2D(num_outputs[0], 1, 1, border_mode='same',activation='relu')(input)
  ince_3x3_red = Convolution2D(num_outputs[1], 1, 1, border_mode='same',activation='relu')(input)
  ince_3x3 = Convolution2D(num_outputs[2], 3, 3, border_mode='same', activation='relu')(ince_3x3_red)
  ince_5x5_red = Convolution2D(num_outputs[3], 1, 1, border_mode='same', activation='relu')(input)
  ince_5x5 = Convolution2D(num_outputs[4], 5, 5, border_mode='same', activation='relu')(ince_5x5_red)
  ince_pool = MaxPooling2D(3,1,border_mode='same')(input)
  ince_pool_proj = Convolution2D(num_outputs[5], 1, 1, border_mode='same', activation='relu')(ince_pool)
  
  ince_out = merge([ince_1x1, ince_3x3, ince_5x5, ince_pool_proj], mode='concat', concat_axis=-1)
  return ince_out

def InceptionBlock_3(input, num_outputs):
  ince_1x1 = Convolution2D(num_outputs[0], 1, 1, border_mode='same',activation='relu')(input)
  ince_3x3_red = Convolution2D(num_outputs[1], 1, 1, border_mode='same',activation='relu')(input)
  ince_3x3 = Convolution2D(num_outputs[2], 3, 3, border_mode='same', activation='relu')(ince_3x3_red)
  ince_5x5_red = Convolution2D(num_outputs[3], 1, 1, border_mode='same', activation='relu')(input)
  ince_5x5 = Convolution2D(num_outputs[4], 5, 5, border_mode='same', activation='relu')(ince_5x5_red)
  ince_pool = MaxPooling2D(3,1,border_mode='same')(input)
  ince_pool_proj = Convolution2D(num_outputs[5], 1, 1, border_mode='same', activation='relu')(ince_pool)
  
  ince_out = merge([ince_1x1, ince_3x3, ince_5x5, ince_pool_proj], mode='concat', concat_axis=-1)
  return ince_out

def Inception(num_classes=1000, input_shape=(224,224,3)):
  img_input = Input(shape=input_shape)
  
  conv = Convolution2D(32, 3, strides=2)(img_input)
  conv = Convolution2D(32, 3, strides=1)(conv)
  conv = Convolution2D(64, 3, padding='same')(conv)

  pool = MaxPooling2D(3,strides=2)(conv)
  
  conv = Convolution2D(80, 3, strides=2)(img_input)
  conv = Convolution2D(192, 3, strides=1)(conv)
  conv = Convolution2D(288, 3, padding='same')(conv)

  inception = InceptionBlock_1(conv, [96, 96, 96, 96, 144, 48])
  inception = InceptionBlock_1(inception, [96, 96, 96, 96, 144, 48])
  pool = MaxPooling2D(3,strides=2)(inception)

  inception = InceptionBlock_1(pool, [96, 96, 96, 96, 144, 48])
  inception = InceptionBlock_1(inception, [96, 96, 96, 96, 144, 48])
  pool = MaxPooling2D(3,strides=2)(inception)

  inception = InceptionBlock_1(pool, [96, 96, 96, 96, 144, 48])
  inception = InceptionBlock_1(inception, [96, 96, 96, 96, 144, 48])
  pool = MaxPooling2D(3,strides=2)(inception)

  stack = GlobalAveragePooling2D()(pool)
  stack = Dropout(0.4)(stack)
  out = Dense(num_classes)(stack)
  
  model = Model(img_input, out)
  model.compile(loss='categorical_crossentropy', optimizer='sgd')
  return model
 
model = Inception()
print (model.summary())
plot_model(model,'inception.png')
