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







def initial(input, num_output):
  a = Convolution2D(num_output-1, 2, strides=2)(input)
  b = MaxPooling2D(2)(input)
  out =  merge([a,b], mode='concat', concat_axis=-1)
  return out

def bottleneck_en(inp, output, internal_scale=4, asymmetric=0, dilated=0, downsample=False, dropout_rate=0.1):
    # main branch
    internal = output // internal_scale
    encoder = inp

    # 1x1
    input_stride = 2 if downsample else 1  # the 1st 1x1 projection is replaced with a 2x2 convolution when downsampling
    encoder = Conv2D(internal, (input_stride, input_stride),
                            # padding='same',
                            strides=(input_stride, input_stride), use_bias=False)(encoder)
    # Batch normalization + PReLU
    encoder = BatchNormalization(momentum=0.1)(encoder)  # enet uses momentum of 0.1, keras default is 0.99
    encoder = PReLU(shared_axes=[1, 2])(encoder)

    # conv
    if not asymmetric and not dilated:
        encoder = Conv2D(internal, (3, 3), padding='same')(encoder)
    elif asymmetric:
        encoder = Conv2D(internal, (1, asymmetric), padding='same', use_bias=False)(encoder)
        encoder = Conv2D(internal, (asymmetric, 1), padding='same')(encoder)
    elif dilated:
        encoder = Conv2D(internal, (3, 3), dilation_rate=(dilated, dilated), padding='same')(encoder)
    else:
        raise(Exception('You shouldn\'t be here'))

    encoder = BatchNormalization(momentum=0.1)(encoder)  # enet uses momentum of 0.1, keras default is 0.99
    encoder = PReLU(shared_axes=[1, 2])(encoder)
    
    # 1x1
    encoder = Conv2D(output, (1, 1), use_bias=False)(encoder)

    encoder = BatchNormalization(momentum=0.1)(encoder)  # enet uses momentum of 0.1, keras default is 0.99
    encoder = SpatialDropout2D(dropout_rate)(encoder)

    other = inp
    # other branch
    if downsample:
        other = MaxPooling2D()(other)

        other = Permute((1, 3, 2))(other)
        pad_feature_maps = output - inp.get_shape().as_list()[3]
        tb_pad = (0, 0)
        lr_pad = (0, pad_feature_maps)
        other = ZeroPadding2D(padding=(tb_pad, lr_pad))(other)
        other = Permute((1, 3, 2))(other)

    encoder = add([encoder, other])
    encoder = PReLU(shared_axes=[1, 2])(encoder)
    return encoder

def bottleneck_de(encoder, output, upsample=False, reverse_module=False):
    internal = output // 4

    x = Conv2D(internal, (1, 1), use_bias=False)(encoder)
    x = BatchNormalization(momentum=0.1)(x)
    x = Activation('relu')(x)
    if not upsample:
        x = Conv2D(internal, (3, 3), padding='same', use_bias=True)(x)
    else:
        x = Conv2DTranspose(filters=internal, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = BatchNormalization(momentum=0.1)(x)
    x = Activation('relu')(x)

    x = Conv2D(output, (1, 1), padding='same', use_bias=False)(x)

    other = encoder
    if encoder.get_shape()[-1] != output or upsample:
        other = Conv2D(output, (1, 1), padding='same', use_bias=False)(other)
        other = BatchNormalization(momentum=0.1)(other)
        if upsample and reverse_module is not False:
            other = UpSampling2D(size=(2, 2))(other)
        
    if upsample and reverse_module is False:
        decoder = x
    else:
        x = BatchNormalization(momentum=0.1)(x)
        decoder = add([x, other])
        decoder = Activation('relu')(decoder)

    return decoder




def get_model(input_shape=(224,224,3), num_classes=21):
  dropout_rate=0.1
  img_input = Input(shape=input_shape)
  init = initial(img_input,16)
  enet = BatchNormalization(momentum=0.1)(init) 
  enet = PReLU(shared_axes=[1, 2])(enet)
  enet = bottleneck_en(enet, 64, downsample=True, dropout_rate=dropout_rate)  
  for _ in range(4):
    enet = bottleneck_en(enet, 64, dropout_rate=dropout_rate)  
    
  enet = bottleneck_en(enet, 128, downsample=True) 

  for _ in range(2):
    enet = bottleneck_en(enet, 128)  
    enet = bottleneck_en(enet, 128, dilated=2) 
    enet = bottleneck_en(enet, 128, asymmetric=5)  
    enet = bottleneck_en(enet, 128, dilated=4)  
    enet = bottleneck_en(enet, 128) 
    enet = bottleneck_en(enet, 128, dilated=8)  
    enet = bottleneck_en(enet, 128, asymmetric=5) 
    enet = bottleneck_en(enet, 128, dilated=16)

  enet = bottleneck_de(enet, 64, upsample=True, reverse_module=True) 
  enet = bottleneck_de(enet, 64) 
  enet = bottleneck_de(enet, 64)  
  enet = bottleneck_de(enet, 16, upsample=True, reverse_module=True) 
  enet = bottleneck_de(enet, 16) 

  enet = Conv2DTranspose(filters=num_classes, kernel_size=(2, 2), strides=(2, 2), padding='same')(enet)
  model = Model(img_input, enet)
  return model


model = get_model()
print(model.summary())
plot_model(model,'ENet.png')
