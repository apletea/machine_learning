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


def StackBlock(input, num_outputs):
  ince_1x1 = Convolution2D(num_outputs[0], 1, 1, border_mode='same',activation='relu')(input)
  ince_3x3_red = Convolution2D(num_outputs[1], 1, 1, border_mode='same',activation='relu')(input)
  ince_3x3 = Convolution2D(num_outputs[2], 3, 3, border_mode='same', activation='relu')(ince_3x3_red)
  ince_5x5_red = Convolution2D(num_outputs[3], 1, 1, border_mode='same', activation='relu')(input)
  ince_5x5 = Convolution2D(num_outputs[4], 5, 5, border_mode='same', activation='relu')(ince_5x5_red)
  ince_pool = MaxPooling2D(3,1,border_mode='same')(input)
  ince_pool_proj = Convolution2D(num_outputs[5], 1, 1, border_mode='same', activation='relu')(ince_pool)
  
  ince_out = merge([ince_1x1, ince_3x3, ince_5x5, ince_pool_proj], mode='concat', concat_axis=-1)
  return ince_out

def GoogLeNet(num_classes=1000, input_shape=(224,224,3)):
  img_input = Input(shape=input_shape)
  
  conv1 = Convolution2D(64,7,strides=2)(img_input)
  conv1 = ZeroPadding2D(padding=(1, 1))(conv1)
  conv1 = MaxPooling2D(3,strides=2)(conv1)
  
  conv2 = Convolution2D(192,3,strides=2)(conv1)
  conv2 = MaxPooling2D(2,strides=2)(conv2)
  
  stack = StackBlock(conv2, [64, 96, 128, 16, 32, 32])
  stack = StackBlock(stack, [128, 128, 192, 32, 96, 64])
  
  stack = MaxPooling2D(3,strides=2)(stack)
  stack = ZeroPadding2D(padding=(1,1))(stack)
  
  stack = StackBlock(stack, [192, 96, 208, 16, 48, 64])
  stack = StackBlock(stack, [160, 112, 224, 24, 64, 64])
  stack = StackBlock(stack, [128, 128, 256, 24, 64, 64])
  stack = StackBlock(stack, [112, 144, 288, 32, 64, 64])
  stack = StackBlock(stack, [256, 160, 320, 32, 128, 128])

  stack = MaxPooling2D(3, strides=2)(stack)
  
  stack = StackBlock(stack, [256, 160, 320, 32, 128, 128])
  stack = StackBlock(stack, [384, 192, 384, 48, 128, 128])
  
  stack = GlobalAveragePooling2D()(stack)
  stack = Dropout(0.4)(stack)
  out = Dense(num_classes)(stack)
  
  model = Model(img_input, out)
  model.compile(loss='categorical_crossentropy', optimizer='sgd')
  return model  
   
model = GoogLeNet()
print (model.summary())
plot_model(model, to_file='googlenet.png')
