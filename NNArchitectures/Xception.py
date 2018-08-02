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

def MiddleFlow(input):
  a = SeparableConv2D(728, 3, activation='relu', padding='same')(input)
  a = SeparableConv2D(728, 3, activation='relu', padding='same')(a)
  a = SeparableConv2D(728, 3, activation='relu', padding='same')(a)
  out = Add()([a,input])
  return out

def EntryFlow(input, num_output):
  a = SeparableConv2D(num_output, 3, activation='relu', padding='same')(input)
  a = SeparableConv2D(num_output, 3, padding='same')(a)
  a = ZeroPadding2D(padding=(1, 1))(a)
  a = MaxPooling2D(3, strides=2)(a)


  b = Convolution2D(num_output, 1, strides=2)(input)
  out = Add()([a,b])
  return out
 
def Xception(num_classes=1000, input_shape=(299, 299, 3)):
  img_input = Input(shape=input_shape)
  
  conv = Convolution2D(32,3,strides=2, activation='relu')(img_input)
  conv = Convolution2D(64,3, activation='relu')(conv)
  
  outs = [128, 256, 728]
  for i in range(3):
    conv = EntryFlow(conv,outs[i])
  
  for i in range(8):
    conv =  MiddleFlow(conv)
  
  conv = EntryFlow(conv,728)
  
  conv = SeparableConv2D(1536, 3, activation='relu')(conv)
  conv = SeparableConv2D(2048, 3, activation='relu')(conv)
  out = GlobalAveragePooling2D()(conv)
  model = Model(img_input, out)
  model.compile(loss='categorical_crossentropy', optimizer='sgd')
  return model

model = Xception()
print (model.summary())
plot_model(model, 'xception.png')
