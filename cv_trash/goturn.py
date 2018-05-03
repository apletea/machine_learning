import numpy as np
import pandas as pd
import tensorflow as tf
import keras

config = tf.ConfigProto( device_count = {'GPU' : 1, 'CPU':6})
sess = tf.Session(config=config)
keras.backend.set_session(sess)

import cv2

from keras.layers import *
from keras.models import *
from keras.applications import *
from keras.optimizers import *
from keras.regularizers import *
import os

import matplotlib.pyplot as plt

im_size = 227 
chanels = 3
def get_model():
    target_in = Input(shape=(im_size,im_size,chanels))
    img_in    = Input(shape=(im_size,im_size,chanels))
    model_a = ResNet50(include_top=False,weights=None)
    model_b = ResNet50(include_top=False,weights=None)
    target_features = model_a(target_in)
    img_features    = model_b(img_in)
    tmp = Concatenate([target_features,img_features],axis=0)()
#    tmp = Dense(256)(tmp)
#    tmp = Dense(4)(tmp)
    model = Model([target_in,img_in],tmp)
    return model

model = get_model()
model.summary()
