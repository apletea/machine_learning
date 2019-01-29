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

l2_regularizer = tf.contrib.layers.l2_regularizer
leaky_relu = tf.nn.leaky_relu 

model = load_model('inference_fscore_Xception_1.5kkk.hdf5')
i=0



def attach_attention_module(net, attention_module,num):
    if attention_module == 'se_block': # SE_block
        net = se_block(net)
    elif attention_module == 'cbam_block': # CBAM_block
        net = cbam_block(net,8,num)
    else:
        raise Exception("'{}' is not supported attention module!".format(attention_module))

    return net

def cbam_block(cbam_feature, ratio=8,num=0):
    """Contains the implementation of Convolutional Block Attention Module(CBAM) block.
    As described in https://arxiv.org/abs/1807.06521.
    """
    
    cbam_feature = channel_attention(cbam_feature, ratio, num)
    cbam_feature = spatial_attention(cbam_feature, num)
    return cbam_feature

class LRTensorBoard(TensorBoard):
    def __init__(self, log_dir):  # add other arguments to __init__ if you need
        super(LRTensorBoard, self).__init__(log_dir=log_dir)

    def on_epoch_end(self, epoch, logs=None):
        logs.update({'lr': K.eval(self.model.optimizer.lr)})
        super(LRTensorBoard, self).on_epoch_end(epoch, logs)

num=0
def channel_attention(input_feature, ratio=8, num = 0):
    
    channel_axis = 1 if K.image_data_format() == "channels_first"  else -1
    channel = input_feature._keras_shape[channel_axis]
    
    shared_layer_one = Dense(channel//ratio,
        activation='relu',
        kernel_initializer='he_normal',
        use_bias=True,
        bias_initializer='zeros', name=str(num) + 'Dense')
    shared_layer_two = Dense(channel,
        kernel_initializer='he_normal',
        use_bias=True,
        bias_initializer='zeros', name=str(num) + '_Dense')

    avg_pool = GlobalAveragePooling2D(name=str(num) + 'GlobalPolling')(input_feature)    
    avg_pool = Reshape((1,1,channel))(avg_pool)
    avg_pool = Lambda(lambda layer : shared_mlp(
            layer=layer,
            reduction_ratio=ratio,
            training_flag=True,
            l2_regularizer=l2_regularizer,
            leaky_relu=leaky_relu,
            mode=mode
        ))(avg_pool)

    max_pool = GlobalMaxPooling2D(name=str(num) + 'GlobalMaxPooling')(input_feature)
    max_pool = Reshape((1,1,channel))(max_pool)
    max_pool = Lambda(lambda layer : shared_mlp(
            layer=layer,
            reduction_ratio=ratio,
            training_flag=True,
            l2_regularizer=l2_regularizer,
            leaky_relu=leaky_relu,
            mode=mode
        ))(max_pool)
    
    cbam_feature = Add(name=str(num) + 'Add')([avg_pool,max_pool])
    cbam_feature = Activation('sigmoid',name=str(num)+'activ')(cbam_feature)

    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)

    return multiply([input_feature, cbam_feature])

def spatial_attention(input_feature,num):
    kernel_size = 7

    if K.image_data_format() == "channels_first":
        channel = input_feature._keras_shape[1]
        cbam_feature = Permute((2,3,1))(input_feature)
    else:
        channel = input_feature._keras_shape[-1]
    cbam_feature = input_feature
    
    avg_pool = Lambda(lambda x:  tf.reduce_mean(input_tensor=x, axis = 3, keepdims=True ))(cbam_feature)

    max_pool = Lambda(lambda x:  tf.reduce_max(input_tensor=x, axis = 3, keepdims=True ))(cbam_feature)

    concat = Concatenate(axis=3)([avg_pool, max_pool])
    cbam_feature = Conv2D(filters = 1,
        kernel_size=kernel_size,
        strides=1,
        padding='same',
        activation='sigmoid',
        kernel_initializer='he_normal',
        use_bias=False, name=str( num) + 'conv2d_1')(concat)	
    num+=1
    
    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)

    return multiply([input_feature, cbam_feature])


def pain_model(model):
    input = Input(shape=(224,224,3))
    
    block1_conv1 = model.get_layer('block1_conv1')(input)
    block1_conv1_bn = model.get_layer('block1_conv1_bn')(block1_conv1)
    block1_conv1_act = model.get_layer('block1_conv1_act')(block1_conv1_bn)
    block1_conv2 = model.get_layer('block1_conv2')(block1_conv1_act)
    block1_conv2_bn = model.get_layer('block1_conv2_bn')(block1_conv2)
    block1_conv2_act = model.get_layer('block1_conv2_act')(block1_conv2_bn)
    block2_sepconv1 = model.get_layer('block2_sepconv1')(block1_conv2_act)
    block2_sepconv1_bn = model.get_layer('block2_sepconv1_bn')(block2_sepconv1)
    block2_sepconv2_act = model.get_layer('block2_sepconv2_act')(block2_sepconv1_bn)
    block2_sepconv2 = model.get_layer('block2_sepconv2')(block2_sepconv2_act)
    block2_sepconv2_bn = model.get_layer('block2_sepconv2_bn')(block2_sepconv2)
    conv2d_1 = model.get_layer('conv2d_1')(block1_conv2_act)
    block2_pool = model.get_layer('block2_pool')(block2_sepconv2_bn)
    batch_normalization_1 = model.get_layer('batch_normalization_1')(conv2d_1)
    add_1 = model.get_layer('add_1')([batch_normalization_1,block2_pool])   
#    add_1 = attach_attention_module(add_1,'cbam_block',0)

    block3_sepconv1_act = model.get_layer('block3_sepconv1_act')(add_1)
    block3_sepconv1 = model.get_layer('block3_sepconv1')(block3_sepconv1_act)
    block3_sepconv1_bn = model.get_layer('block3_sepconv1_bn')(block3_sepconv1)
    block3_sepconv2_act = model.get_layer('block3_sepconv2_act')(block3_sepconv1_bn)
    block3_sepconv2 = model.get_layer('block3_sepconv2')(block3_sepconv2_act)
    block3_sepconv2_bn = model.get_layer('block3_sepconv2_bn')(block3_sepconv2)
    conv2d_2 = model.get_layer('conv2d_2')(add_1)
    block3_pool = model.get_layer('block3_pool')(block3_sepconv2_bn)
    batch_normalization_2 = model.get_layer('batch_normalization_2')(conv2d_2)
    add_2 = model.get_layer('add_2')([batch_normalization_2,block3_pool])
#    add_2 = attach_attention_module(add_2,'cbam_block',1)

    block4_sepconv1_act = model.get_layer('block4_sepconv1_act')(add_2)
    block4_sepconv1 = model.get_layer('block4_sepconv1')(block4_sepconv1_act)
    block4_sepconv1_bn = model.get_layer('block4_sepconv1_bn')(block4_sepconv1)
    block4_sepconv2_act = model.get_layer('block4_sepconv2_act')(block4_sepconv1_bn)
    block4_sepconv2 = model.get_layer('block4_sepconv2')(block4_sepconv2_act)
    block4_sepconv2_bn = model.get_layer('block4_sepconv2_bn')(block4_sepconv2)
    conv2d_3 = model.get_layer('conv2d_3')(add_2)
    block4_pool = model.get_layer('block4_pool')(block4_sepconv2_bn)
    batch_normalization_3 = model.get_layer('batch_normalization_3')(conv2d_3)
    add_3 = model.get_layer('add_3')([batch_normalization_3, block4_pool])
#    add_3 = attach_attention_module(add_3,'cbam_block',2)
 
    block5_sepconv1_act = model.get_layer('block5_sepconv1_act')(add_3)
    block5_sepconv1 = model.get_layer('block5_sepconv1')(block5_sepconv1_act)
    block5_sepconv1_bn = model.get_layer('block5_sepconv1_bn')(block5_sepconv1)
    block5_sepconv2_act = model.get_layer('block5_sepconv2_act')(block5_sepconv1_bn)
    block5_sepconv2 = model.get_layer('block5_sepconv2')(block5_sepconv2_act)
    block5_sepconv2_bn = model.get_layer('block5_sepconv2_bn')(block5_sepconv2)
    block5_sepconv3_act = model.get_layer('block5_sepconv3_act')(block5_sepconv2_bn)
    block5_sepconv3 = model.get_layer('block5_sepconv3')(block5_sepconv3_act)
    block5_sepconv3_bn = model.get_layer('block5_sepconv3_bn')(block5_sepconv3)
    add_4 = model.get_layer('add_4')([block5_sepconv3_bn, add_3])
#    add_4 = attach_attention_module(add_4,'cbam_block',3)

    block6_sepconv1_act = model.get_layer('block6_sepconv1_act')(add_4)
    block6_sepconv1 = model.get_layer('block6_sepconv1')(block6_sepconv1_act)
    block6_sepconv1_bn = model.get_layer('block6_sepconv1_bn')(block6_sepconv1)
    block6_sepconv2_act = model.get_layer('block6_sepconv2_act')(block6_sepconv1_bn)
    block6_sepconv2 = model.get_layer('block6_sepconv2')(block6_sepconv2_act)
    block6_sepconv2_bn = model.get_layer('block6_sepconv2_bn')(block6_sepconv2)
    block6_sepconv3_act = model.get_layer('block6_sepconv3_act')(block6_sepconv2_bn)
    block6_sepconv3 = model.get_layer('block6_sepconv3')(block6_sepconv3_act)
    block6_sepconv3_bn = model.get_layer('block6_sepconv3_bn')(block6_sepconv3)
    add_5 = model.get_layer('add_5')([block6_sepconv3_bn, add_4])
#    add_5 = attach_attention_module(add_5,'cbam_block',4)

    block7_sepconv1_act = model.get_layer('block7_sepconv1_act')(add_5)
    block7_sepconv1 = model.get_layer('block7_sepconv1')(block7_sepconv1_act)
    block7_sepconv1_bn = model.get_layer('block7_sepconv1_bn')(block7_sepconv1)
    block7_sepconv2_act = model.get_layer('block7_sepconv2_act')(block7_sepconv1_bn)
    block7_sepconv2 = model.get_layer('block7_sepconv2')(block7_sepconv2_act)
    block7_sepconv2_bn = model.get_layer('block7_sepconv2_bn')(block7_sepconv2)
    block7_sepconv3_act = model.get_layer('block7_sepconv3_act')(block7_sepconv2_bn)
    block7_sepconv3 = model.get_layer('block7_sepconv3')(block7_sepconv3_act)
    block7_sepconv3_bn = model.get_layer('block7_sepconv3_bn')(block7_sepconv3)
    add_6 = model.get_layer('add_6')([block7_sepconv3_bn,add_5])
#    add_6 = attach_attention_module(add_6,'cbam_block',5)

    block8_sepconv1_act = model.get_layer('block8_sepconv1_act')(add_6)
    block8_sepconv1 = model.get_layer('block8_sepconv1')(block8_sepconv1_act)
    block8_sepconv1_bn = model.get_layer('block8_sepconv1_bn')(block8_sepconv1)
    block8_sepconv2_act = model.get_layer('block8_sepconv2_act')(block8_sepconv1_bn)
    block8_sepconv2 = model.get_layer('block8_sepconv2')(block8_sepconv2_act)
    block8_sepconv2_bn = model.get_layer('block8_sepconv2_bn')(block8_sepconv2)
    block8_sepconv3_act = model.get_layer('block8_sepconv3_act')(block8_sepconv2_bn)
    block8_sepconv3 = model.get_layer('block8_sepconv3')(block8_sepconv3_act)
    block8_sepconv3_bn = model.get_layer('block8_sepconv3_bn')(block8_sepconv3)
    add_7 = model.get_layer('add_7')([block8_sepconv3_bn, add_6])
#    add_7 = attach_attention_module(add_7,'cbam_block',6)
 
    block9_sepconv1_act = model.get_layer('block9_sepconv1_act')(add_7)
    block9_sepconv1 = model.get_layer('block9_sepconv1')(block9_sepconv1_act)
    block9_sepconv1_bn = model.get_layer('block9_sepconv1_bn')(block9_sepconv1)
    block9_sepconv2_act = model.get_layer('block9_sepconv2_act')(block9_sepconv1_bn)
    block9_sepconv2 = model.get_layer('block9_sepconv2')(block9_sepconv2_act)
    block9_sepconv2_bn = model.get_layer('block9_sepconv2_bn')(block9_sepconv2)
    block9_sepconv3_act = model.get_layer('block9_sepconv3_act')(block9_sepconv2_bn)
    block9_sepconv3 = model.get_layer('block9_sepconv3')(block9_sepconv3_act)
    block9_sepconv3_bn = model.get_layer('block9_sepconv3_bn')(block9_sepconv3)
    add_8 = model.get_layer('add_8')([block9_sepconv3_bn, add_7])
#    add_8 = attach_attention_module(add_8,'cbam_block',7)

    block10_sepconv1_act = model.get_layer('block10_sepconv1_act')(add_8)
    block10_sepconv1 = model.get_layer('block10_sepconv1')(block10_sepconv1_act)
    block10_sepconv1_bn = model.get_layer('block10_sepconv1_bn')(block10_sepconv1)
    block10_sepconv2_act = model.get_layer('block10_sepconv2_act')(block10_sepconv1_bn)
    block10_sepconv2 = model.get_layer('block10_sepconv2')(block10_sepconv2_act)
    block10_sepconv2_bn = model.get_layer('block10_sepconv2_bn')(block10_sepconv2)
    block10_sepconv3_act = model.get_layer('block10_sepconv3_act')(block10_sepconv2_bn)
    block10_sepconv3 = model.get_layer('block10_sepconv3')(block10_sepconv3_act)
    block10_sepconv3_bn = model.get_layer('block10_sepconv3_bn')(block10_sepconv3)
    add_9 = model.get_layer('add_9')([block10_sepconv3_bn, add_8])
#    add_9 = attach_attention_module(add_9,'cbam_block',8)

    block11_sepconv1_act = model.get_layer('block11_sepconv1_act')(add_9)
    block11_sepconv1 = model.get_layer('block11_sepconv1')(block11_sepconv1_act)
    block11_sepconv1_bn = model.get_layer('block11_sepconv1_bn')(block11_sepconv1)
    block11_sepconv2_act = model.get_layer('block11_sepconv2_act')(block11_sepconv1_bn)
    block11_sepconv2 = model.get_layer('block11_sepconv2')(block11_sepconv2_act)
    block11_sepconv2_bn = model.get_layer('block11_sepconv2_bn')(block11_sepconv2)
    block11_sepconv3_act = model.get_layer('block11_sepconv3_act')(block11_sepconv2_bn)
    block11_sepconv3 = model.get_layer('block11_sepconv3')(block11_sepconv3_act)
    block11_sepconv3_bn = model.get_layer('block11_sepconv3_bn')(block11_sepconv3)
    add_10 = model.get_layer('add_10')([block11_sepconv3_bn, add_9])
#    add_10 = attach_attention_module(add_10,'cbam_block',9)

    block12_sepconv1_act = model.get_layer('block12_sepconv1_act')(add_10)
    block12_sepconv1 = model.get_layer('block12_sepconv1')(block12_sepconv1_act)
    block12_sepconv1_bn = model.get_layer('block12_sepconv1_bn')(block12_sepconv1)
    block12_sepconv2_act = model.get_layer('block12_sepconv2_act')(block12_sepconv1_bn)
    block12_sepconv2 = model.get_layer('block12_sepconv2')(block12_sepconv2_act)
    block12_sepconv2_bn = model.get_layer('block12_sepconv2_bn')(block12_sepconv2)
    block12_sepconv3_act = model.get_layer('block12_sepconv3_act')(block12_sepconv2_bn)
    block12_sepconv3 = model.get_layer('block12_sepconv3')(block12_sepconv3_act)
    block12_sepconv3_bn = model.get_layer('block12_sepconv3_bn')(block12_sepconv3)
    add_11 = model.get_layer('add_11')([block12_sepconv3_bn, add_10])
#    add_11 = attach_attention_module(add_11,'cbam_block',10)

    block13_sepconv1_act = model.get_layer('block13_sepconv1_act')(add_11)
    block13_sepconv1 = model.get_layer('block13_sepconv1')(block13_sepconv1_act)
    block13_sepconv1_bn = model.get_layer('block13_sepconv1_bn')(block13_sepconv1)
    block13_sepconv2_act = model.get_layer('block13_sepconv2_act')(block13_sepconv1_bn)
    block13_sepconv2 = model.get_layer('block13_sepconv2')(block13_sepconv2_act)
    block13_sepconv2_bn = model.get_layer('block13_sepconv2_bn')(block13_sepconv2)
    conv2d_4 = model.get_layer('conv2d_4')(add_11)
    block13_pool = model.get_layer('block13_pool')(block13_sepconv2_bn)
    batch_normalization_4 = model.get_layer('batch_normalization_4')(conv2d_4)
    add_12 = model.get_layer('add_12')([batch_normalization_4,block13_pool])
#    add_12 = attach_attention_module(add_12,'cbam_block',11)

    block14_sepconv1 = model.get_layer('block14_sepconv1')(add_12)
    block14_sepconv1_bn = model.get_layer('block14_sepconv1_bn')(block14_sepconv1)
    block14_sepconv1_act = model.get_layer('block14_sepconv1_act')(block14_sepconv1_bn)
    block14_sepconv2 = model.get_layer('block14_sepconv2')(block14_sepconv1_act)
    block14_sepconv2_bn = model.get_layer('block14_sepconv2_bn')(block14_sepconv2)
    block14_sepconv2_act = model.get_layer('block14_sepconv2_act')(block14_sepconv2_bn)
    global_average_pooling2d_1 = model.get_layer('global_average_pooling2d_1')(block14_sepconv2_act)
    dense_13 = model.get_layer('dense_7')(global_average_pooling2d_1)
    ip1 = model.get_layer('ip1')(dense_13)
    dense_14 = model.get_layer('dense_8')(ip1)
    ip2 = model.get_layer('activation_4')(dense_14)
    
    
    return Model(input, ip2)

new_model = pain_model(model)
#new_model.compile(loss='categorical_crossentropy', optimizer='adam')
print (new_model.summary())
#callbacks = [LRTensorBoard(log_dir='./logidr')]
#new_model.fit(np.zeros((1,224,224,3)),np.zeros((1,2)), callbacks=callbacks)
new_model.save('py3_tf_xce.hdf5')
