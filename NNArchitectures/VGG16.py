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

def VGG16(num_classes=1000, input_shape=(224,224,3)):
  model = Sequential()
  model.add(Convolution2D(64,3,padding='same',input_shape=input_shape))
  model.add(Convolution2D(64,3,padding='same'))
  model.add(MaxPooling2D(2,strides=2))
  
  model.add(Convolution2D(128,3, padding='same'))
  model.add(Convolution2D(128,3, padding='same'))
  model.add(MaxPooling2D(2,strides=2))
  
  model.add(Convolution2D(256,3,padding='same'))
  model.add(Convolution2D(256,3,padding='same'))
  model.add(Convolution2D(256,3,padding='same'))
  model.add(MaxPooling2D(2,strides=2))

  
  model.add(Convolution2D(512,3,padding='same'))
  model.add(Convolution2D(512,3,padding='same'))
  model.add(Convolution2D(512,3,padding='same'))
  model.add(Convolution2D(512,3,padding='same'))
  model.add(MaxPooling2D(2,strides=2))

  model.add(Convolution2D(512,3,padding='same'))
  model.add(Activation('relu'))
  model.add(BatchNormalization())
  model.add(Convolution2D(512,3,padding='same'))
  model.add(Activation('relu'))
  model.add(BatchNormalization())
  model.add(Convolution2D(512,3,padding='same'))
  model.add(Activation('relu'))
  model.add(BatchNormalization())
  model.add(MaxPooling2D(14))
  
  model.add(Dense(4096))
  model.add(Activation('relu'))
  model.add(BatchNormalization())
  
  model.add(Dense(4096))
  model.add(Activation('relu'))
  model.add(BatchNormalization())

  model.add(Dense(num_classes))
  model.add(Activation('softmax'))


  model.compile(loss='categorical_crossentropy', optimizer='sgd')
  return model

model = VGG16()
print (model.summary())

