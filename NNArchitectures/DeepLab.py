import cv2
import numpy as np
import random
import skimage
from skimage import data
#from sklearn.model_selection import train_test_split
import pandas as pd
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

from keras.utils import plot_model

import pydensecrf.densecrf as dcrf


def get_model(input_shape=(224,224,3), num_classes=3):
  img_input = Input(shape=input_shape)
  conv1 = Convolution2D(64,3, padding='same', activation='relu')(img_input)
  conv2 = Convolution2D(64,3, padding='same', activation='relu')(conv1)
  
  pool1 = MaxPooling2D(2, strides=2)(conv2)
  conv3 = Convolution2D(128,3, padding='same', activation='relu')(pool1)
  conv4 = Convolution2D(128,3, padding='same', activation='relu')(conv3)

  pool2 = MaxPooling2D(2, strides=2)(conv4)
  conv5 = Convolution2D(256,3, padding='same', activation='relu')(pool2)
  conv6 = Convolution2D(256,3, padding='same', activation='relu')(conv5)
  conv7 = Convolution2D(256,3, padding='same', activation='relu')(conv6)
  
  pool3 = MaxPooling2D(2, strides=2)(conv7)
  conv8 = Convolution2D(512,3,  padding='same', activation='relu')(pool3)
  conv9 = Convolution2D(512,3,  padding='same', activation='relu')(conv8)
  conv10 = Convolution2D(512,3, padding='same', activation='relu')(conv9)
  
  pool4 = MaxPooling(2, strides=2)(conv10)
  conv11 = Convolution2D(512,3,dilation_rate=2, padding='same', activation='relu')(pool4)
  conv12 = Convolution2D(512,3,dilation_rate=2, padding='same', activation='relu')(conv11)
  conv13 = Convolution2D(512,3,dilation_rate=2, padding='same', activation='relu')(conv12)

  pool5 = MaxPooling(2, strides=2)(conv13)
  covn14 = Convolution2D(4096, 4, dilation_rate=4, padding='same', activation='relu')(pool5) 
  conv15 = Convolution2D(4096, 4, dilation_rate=4, padding='same', activation='relu')(conv14)

  conv16 = Convolution2D(num_classes, 1, padding='same', activation='sigmoid')(conv15)
  

  model = Model(img_input, covn16)
  model.compile(loss='categorical_crossentropy', optimizer='adam')
  return model

def crf(original_image, mask_img):
    
    # Converting annotated image to RGB if it is Gray scale
    if(len(mask_img.shape)<3):
        mask_img = gray2rgb(mask_img)

#     #Converting the annotations RGB color to single 32 bit integer
    annotated_label = mask_img[:,:,0] + (mask_img[:,:,1]<<8) + (mask_img[:,:,2]<<16)
    
#     # Convert the 32bit integer color to 0,1, 2, ... labels.
    colors, labels = np.unique(annotated_label, return_inverse=True)

    n_labels = 2
    
    #Setting up the CRF model
    d = dcrf.DenseCRF2D(original_image.shape[1], original_image.shape[0], n_labels)

    # get unary potentials (neg log probability)
    U = unary_from_labels(labels, n_labels, gt_prob=0.7, zero_unsure=False)
    d.setUnaryEnergy(U)

    # This adds the color-independent term, features are the locations only.
    d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL,
                      normalization=dcrf.NORMALIZE_SYMMETRIC)
        
    #Run Inference for 10 steps 
    Q = d.inference(10)

    # Find out the most probable class for each pixel.
    MAP = np.argmax(Q, axis=0)

    return MAP.reshape((original_image.shape[0],original_image.shape[1]))

model = get_model()
print(model.summary())
plot_model(model, 'deeplabV1.png')
