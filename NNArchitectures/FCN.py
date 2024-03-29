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
from Xception import Xception
def FCN(num_classes=21, input_shape=(224,224,3)):
  base_net = Xception(num_classes, input_shape)
  x =  base_net.layers[-2].output
  conv = Convolution2D(num_classes,1,padding='same')(x)
  up = UpSampling2D((75,75))(conv)
  up = Convolution2D(num_classes,2,padding='valid')(up)
  model = Model(base_net.input,up)
  model.compile(loss='binary_crossentropy', optimizer='adam')
  return model



model = FCN()
print (model.summary())
plot_model(model, 'FCN.png')
