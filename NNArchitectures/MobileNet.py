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

def ConvBnRelu(input, num_output, padd='valid', stride = 1, filtr=3):
  x = Convolution2D(num_output, filtr, strides=stride, padding=padd)(input)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  return x

def DepthWiseConv(input, num_output, padd='valid', stride = 1):
  x = SeparableConv2D(num_output, 3, activation='relu', padding=padd, strides=stride)(input)
  x = BatchNormalization()(x)
  return x

def MobileNet(num_classes=1000, input_shape=(224,224,3)):
  img_input = Input(shape=input_shape)
  
  x = ConvBnRelu(img_input,32,stride = 2)
  x = DepthWiseConv(x, 32, padd='same')
  x = ConvBnRelu(x, 64, padd='same', filtr=1)
  x = DepthWiseConv(x, 64,stride=2)
  x = ConvBnRelu(x, 128, padd='same', filtr=1)
  x = DepthWiseConv(x, 128, padd='same')
  x = ConvBnRelu(x, 128, padd='same', filtr=1)
  x = DepthWiseConv(x, 256, stride=2)
  x = ConvBnRelu(x, 256, padd='same', filtr=1)
  x = DepthWiseConv(x, 256, padd='same')
  x = ConvBnRelu(x, 256, padd='same', filtr=1)
  x = DepthWiseConv(x, 256, stride=2)
  x = ConvBnRelu(x, 512, padd='same', filtr=1)

  for i in range(5):
    x = DepthWiseConv(x, 512, padd='same')
    x = ConvBnRelu(x, 512, padd='same', filtr=1)  
  x = DepthWiseConv(x, 512, stride=2)
  x = ConvBnRelu(x, 1024, padd='same', filtr=1)
  out = GlobalAveragePooling2D()(x)
  out = Dense(num_classes)(out)

  model = Model(img_input, out)
  model.compile(loss='categorical_crossentropy', optimizer='sgd')

  return model
model = MobileNet()
print (model.summary())
plot_model(model, 'mobilenet.png')
