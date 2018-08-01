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
def Residual(input, num_output=64):
  
  output = Convolution2D(num_output, 1, padding = 'valid')(input)
  output = BatchNormalization()(output)
  output = Activation('relu')(output)
  output = Convolution2D(num_output, 3, strides = 2)(output)
  output = BatchNormalization()(output)
  output = Activation('relu')(output)
  
  return output
  

def Identity(input, num_output=64):

  output = Convolution2D(num_output, 3, padding='same')(input)
  output = BatchNormalization()(output)
  output = Activation('relu')(output)
  output = Convolution2D(num_output, 3, padding='same')(output)
  output = BatchNormalization()(output)
  output = Activation('relu')(output)

  return output

def ResNet18(num_classes=1000, input_shape=(224,224,3)):
  img_input = Input(shape=input_shape)
  base_num = 64  
  count = 2

  first = Convolution2D(64,3,strides=2)(img_input)
  first = MaxPooling2D(2, strides=2)(first)
  
  for i in range(4):
    second = first
    for j in range(count):  
      tmp = Identity(second, base_num*(2**i))
      second = Add()([second,tmp])
    if (i == 3):
      break
    first = Residual(second,base_num*(2**(i+1))) 
  
  output = GlobalAveragePooling2D()(second)
  output = Dense(num_classes)(output)
#  output = Activation('softmax')(output)
  
  model = Model(img_input, output)
  model.compile(loss='categorical_crossentropy', optimizer='sgd')
  return model
  

model = ResNet18()
print (model.summary())
plot_model(model, to_file='resnet18.png')
