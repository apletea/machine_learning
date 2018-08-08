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

from keras import backend as K
from keras.layers import Layer


class MaxPoolingWithArgmax2D(Layer):
    def __init__(self, pool_size=(2, 2), strides=(2, 2), padding='same', **kwargs):
        super(MaxPoolingWithArgmax2D, self).__init__(**kwargs)
        self.padding = padding
        self.pool_size = pool_size
        self.strides = strides

    def call(self, inputs, **kwargs):
        padding = self.padding
        pool_size = self.pool_size
        strides = self.strides
        if K.backend() == 'tensorflow':
            ksize = [1, pool_size[0], pool_size[1], 1]
            padding = padding.upper()
            strides = [1, strides[0], strides[1], 1]
            output, argmax = K.tf.nn.max_pool_with_argmax(inputs, ksize=ksize, strides=strides, padding=padding)
        else:
            errmsg = '{} backend is not supported for layer {}'.format(K.backend(), type(self).__name__)
            raise NotImplementedError(errmsg)
        argmax = K.cast(argmax, K.floatx())
        return [output, argmax]

    def compute_output_shape(self, input_shape):
        ratio = (1, 2, 2, 1)
        output_shape = [dim // ratio[idx] if dim is not None else None for idx, dim in enumerate(input_shape)]
        output_shape = tuple(output_shape)
        return [output_shape, output_shape]

    def compute_mask(self, inputs, mask=None):
        return 2 * [None]


class MaxUnpooling2D(Layer):
    def __init__(self, size=(2, 2), **kwargs):
        super(MaxUnpooling2D, self).__init__(**kwargs)
        self.size = size

    def call(self, inputs, output_shape=None):
        updates, mask = inputs[0], inputs[1]
        with K.tf.variable_scope(self.name):
            mask = K.cast(mask, 'int32')
            input_shape = K.tf.shape(updates, out_type='int32')
            #  calculation new shape
            if output_shape is None:
                output_shape = (input_shape[0], input_shape[1] * self.size[0], input_shape[2] * self.size[1], input_shape[3])
            self.output_shape1 = output_shape

            # calculation indices for batch, height, width and feature maps
            one_like_mask = K.ones_like(mask, dtype='int32')
            batch_shape = K.concatenate([[input_shape[0]], [1], [1], [1]], axis=0)
            batch_range = K.reshape(K.tf.range(output_shape[0], dtype='int32'), shape=batch_shape)
            b = one_like_mask * batch_range
            y = mask // (output_shape[2] * output_shape[3])
            x = (mask // output_shape[3]) % output_shape[2]
            feature_range = K.tf.range(output_shape[3], dtype='int32')
            f = one_like_mask * feature_range

            # transpose indices & reshape update values to one dimension
            updates_size = K.tf.size(updates)
            indices = K.transpose(K.reshape(K.stack([b, y, x, f]), [4, updates_size]))
            values = K.reshape(updates, [updates_size])
            ret = K.tf.scatter_nd(indices, values, output_shape)
            return ret

    def compute_output_shape(self, input_shape):
        mask_shape = input_shape[1]
        return mask_shape[0], mask_shape[1] * self.size[0], mask_shape[2] * self.size[1], mask_shape[3]


def double_conv_layer(x, size, dropout, batch_norm):
  if K.image_dim_ordering() == 'th':
    axis = 1
  else:
    axis = 3
  conv = Convolution2D(size, (3, 3), padding='same')(x)
  if batch_norm is True:
    conv = BatchNormalization(axis=axis)(conv)
  conv = Activation('relu')(conv)
  conv = Convolution2D(size, (3, 3), padding='same')(conv)
  if batch_norm is True:
    conv = BatchNormalization(axis=axis)(conv)
  conv = Activation('relu')(conv)
  if dropout > 0:
     conv = Dropout(dropout)(conv)
  return conv

def double_conv_conc(x, size, dropout, batch_norm):
  conv = BatchNormalization()(x)
  conv = Activation('relu')(conv)
  conv = Convolution2D(size, (3, 3), padding='same')(conv)  

def MiddleFlow(input, num_output):
  a = Convolution2D(num_output, 1, activation='relu', padding='same')(input)
  a = BatchNormalization()(a)
  a = SeparableConv2D(num_output, 3, activation='relu', padding='same')(input)
  a = BatchNormalization()(a)
  a = SeparableConv2D(num_output, 3, activation='relu', padding='same')(input)
  a = BatchNormalization()(a)
  a = SeparableConv2D(num_output, 3, activation='relu', padding='same')(input)
  a = BatchNormalization()(a)
  out = Concatenate()([a,input])
  out = Convolution2D(num_output, 1, activation='relu', padding='same')(out)
  out = BatchNormalization()(out)
  return out

def EntryFlow(input, num_output):
  a = SeparableConv2D(num_output, 3, activation='relu', padding='same')(input)
  a = BatchNormalization()(a)
  a = SeparableConv2D(num_output, 3,activation='relu', padding='same')(a)
  a = BatchNormalization()(a)
  a = MaxPooling2D(2)(a)
  b = Convolution2D(num_output, 1, strides=2, activation='relu')(input)
  b = BatchNormalization()(b)
  out = Concatenate()([a,b])
  out = Convolution2D(num_output, 1, activation='relu', padding='same')(out)
  out = BatchNormalization()(out)
  return out

def SegNet(input_shape=(224,224,3)):
  filters = 32
  dropout = False
  batch_norm = True
  img_input = Input(shape=input_shape)
  depth_input = Input(shape=(1,1,1))

  conv112 = double_conv_layer(img_input, filters, dropout, batch_norm)

  pool56 = EntryFlow(conv112, filters*2)
  conv56 = MiddleFlow(pool56, filters*2)
  tmp = Concatenate()([conv56,pool56])
  conv56_1 = MiddleFlow(tmp, filters*2)
  tmp = Concatenate()([conv56_1,conv56,pool56])
  conv56_2 = MiddleFlow(tmp, filters*2)  
  tmp =  Concatenate()([conv56_2,conv56_1,conv56,pool56])

  pool28 = EntryFlow(tmp,  filters*4)
  conv28 = MiddleFlow(pool28, filters*4)
  tmp = Concatenate()([conv28,pool28])
  conv28_1 = MiddleFlow(tmp, filters*4) 
  tmp = Concatenate()([conv28_1,conv28,pool28])
  conv28_2 = MiddleFlow(tmp, filters*4) 
  tmp =  Concatenate()([conv28_2,conv28_1,conv28,pool28])

  pool14 = EntryFlow(tmp,  filters*8)
  conv14 = MiddleFlow(pool14, filters*8)
  tmp = Concatenate()([conv14,pool14])
  conv14_1 = MiddleFlow(tmp, filters*8)
  tmp = Concatenate()([conv14_1,conv14,pool14])
  conv14_2 = MiddleFlow(tmp, filters*8) 
  tmp =  Concatenate()([conv14_2,conv14_1,conv14,pool14])

  
  pool7 = EntryFlow(tmp,   filters*16)
  conv7 = MiddleFlow(pool7, filters*16) 
  tmp = Concatenate()([conv7,pool7])
  conv7_1 = MiddleFlow(tmp, filters*16) 
  tmp = Concatenate()([conv7_1,conv7,pool7])
  conv7_2 = MiddleFlow(tmp, filters*16)
  tmp =  Concatenate()([conv7_2,conv7_1,conv7,pool7])

  pool4 = EntryFlow(tmp,   filters*32)
  conv4 = MiddleFlow(pool4, filters*32) 
  conv4_1 = MiddleFlow(conv4, filters*32) 
  conv4_2 = MiddleFlow(conv4_1, filters*32)
 

  up7 = concatenate([UpSampling2D(size=(2, 2))(conv4_2), conv7], axis=-1)
  conv7_3 = MiddleFlow(up7, filters*16)
  tmp =  Concatenate()([conv7_3,conv7_2])
  conv7_4 = MiddleFlow(tmp, filters*16) 
  tmp =  Concatenate()([conv7_4,conv7_1])
  conv7_5 = MiddleFlow(tmp, filters*16)
  tmp =  Concatenate()([conv7_5,conv7])

  up14 = concatenate([MaxUnpooling2D(size=(2, 2))(tmp), conv14], axis=-1)
  conv14_3 = MiddleFlow(up14, filters*8)
  tmp =  Concatenate()([conv14_3,conv14_2])
  conv14_4 = MiddleFlow(tmp, filters*8) 
  tmp =  Concatenate()([conv14_4,conv14_1])
  conv14_5 = MiddleFlow(tmp, filters*8)
  tmp =  Concatenate()([conv14_5,conv14])

  up28 = concatenate([MaxUnpooling2D(size=(2, 2))(tmp), conv28], axis=-1)
  conv28_3 = MiddleFlow(up28, filters*4)
  tmp =  Concatenate()([conv28_3,conv28_2])
  conv28_4 = MiddleFlow(tmp, filters*4) 
  tmp =  Concatenate()([conv28_4,conv28_1])
  conv28_5 = MiddleFlow(tmp, filters*4) 
  tmp =  Concatenate()([conv28_5,conv28])

  up56 = concatenate([MaxUnpooling2D(size=(2, 2))(tmp), conv56], axis=-1)
  conv56_3 = MiddleFlow(up56, filters*2)
  tmp =  Concatenate()([conv56_3,conv56_2])
  conv56_4 = MiddleFlow(tmp, filters*2)
  tmp =  Concatenate()([conv56_4,conv56_1])
  conv56_5 = MiddleFlow(tmp, filters*2) 
  tmp =  Concatenate()([conv56_5,conv56])

  up112 = concatenate([MaxUnpooling2D(size=(2, 2))(tmp), conv112], axis=-1)
  conv112_1 = MiddleFlow(up112, filters)
  tmp = Concatenate()([conv112_1,conv112])
  conv112 = Convolution2D(num_classes, (1, 1))(up112)


  model = Model(img_input, conv112)
  return model





model = SegNet()
print (model.summary())
plot_model(model,'SegNet.png')
