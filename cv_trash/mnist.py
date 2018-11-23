
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


config = tf.ConfigProto(intra_op_parallelism_threads=8,\
        inter_op_parallelism_threads=8, allow_soft_placement=True,\
        device_count = {'CPU' : 8, 'GPU' : 0})
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
from keras.datasets import mnist

import keras
from sklearn.metrics import roc_auc_score
import sys
import matplotlib.pyplot as plt
from keras.models import Model
import numpy as np


batch_size = 128
num_classes = 10
epochs = 50

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
x_tmp = np.zeros(x_train.shape+(1,))
x_tmp[:,:,:,0] = x_train
x_train = x_tmp

x_tmp = np.zeros(x_test.shape+(1,))
x_tmp[:,:,:,0] = x_test
x_test = x_tmp
# Maintain single value ground truth labels for center loss inputs
# Because Embedding layer only accept index as inputs instead of one-hot vector
y_train_value = y_train
y_test_value = y_test

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

class Histories(keras.callbacks.Callback):
    def __init__(self, isCenterloss, strs):
        self.isCenterloss = isCenterloss
        self.str = strs

    def on_train_begin(self, logs={}):
        self.aucs = []
        self.losses = []

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
		
        print('\n=========')
        print(len(self.validation_data)) #be careful of the dimenstion of the self.validation_data, somehow some extra dim will be included
        print(self.validation_data[0].shape)
        print(self.validation_data[1].shape)
        print('=========')
		#(IMPORTANT) Only use one input: "inputs=self.model.input[0]"
        ip1_input = self.model.input #this can be a list or a matrix. 
        if self.isCenterloss:
            ip1_input = self.model.input[0]
            labels = self.validation_data[1].flatten() # already are single value ground truth labels
        else:
            labels = np.argmax(self.validation_data[1],axis=1) #make one-hot vector to index for visualization
		
        ip1_layer_model = Model(inputs=ip1_input, outputs=self.model.get_layer('ip1').output)
        ip1_output = ip1_layer_model.predict(self.validation_data[0])
		
        visualize(ip1_output,labels,epoch, self.str)
		
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return


def visualize(feat, labels, epoch, substr='softmax'):

    plt.ion()
    c = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff',
         '#ff00ff', '#990000', '#999900', '#009900', '#009999']
    plt.clf()
    for i in range(10):
        plt.plot(feat[labels == i, 0], feat[labels == i, 1], '.', c=c[i])
    plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], loc = 'upper right')
    XMax = np.max(feat[:,0]) 
    XMin = np.min(feat[:,1])
    YMax = np.max(feat[:,0])
    YMin = np.min(feat[:,1])

    plt.xlim(xmin=XMin,xmax=XMax)
    plt.ylim(ymin=YMin,ymax=YMax)
    plt.text(XMin,YMax,"epoch=%d" % epoch)
    plt.savefig('imgs/{}/epoch={}.jpg'.format(substr,str(epoch)))
    plt.draw()
    plt.pause(0.001)



sq1x1 = "squeeze1x1"
exp1x1 = "expand1x1"
exp3x3 = "expand3x3"
relu = "relu_"


btlnek_1 = "btlnek_1"
btlnek_2 = "btlnek_2"
conv1_3 = "conv1_3"
conv3_1 = "conv3_1"

def fire_module(x, fire_id, squeeze=16, expand=64):
    s_id = 'fire' + str(fire_id) + '/'

    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = 3
    
    x = Convolution2D(squeeze, (1, 1), padding='valid', name=s_id + sq1x1)(x)
    x = Activation('relu', name=s_id + relu + sq1x1)(x)

    left = Convolution2D(expand, (1, 1), padding='valid', name=s_id + exp1x1)(x)
    left = Activation('relu', name=s_id + relu + exp1x1)(left)

    right = Convolution2D(expand, (3, 3), padding='same', name=s_id + exp3x3)(x)
    right = Activation('relu', name=s_id + relu + exp3x3)(right)

    x = add([left, right], name=s_id + 'add')
    return x

def fire_next_module(in_p, fire_id, squeeze=16, expand=64, expand2=-1, batch=False):
    s_id = 'fire' + str(fire_id) + '/'
    if (expand2 == -1):
        expand2 = expand
    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = 3
        
    x = Convolution2D(expand, (1, 1), padding='valid', name=s_id + btlnek_1)(in_p)
    if (batch):
        x = BatchNormalization()(x)
    x = Activation('relu', name=s_id + relu + btlnek_1)(x)
    
    x = Convolution2D(squeeze,(1, 1), padding='valid', name=s_id + btlnek_2)(x)
    if (batch):
        x = BatchNormalization()(x)
    x = Activation('relu', name=s_id + relu + btlnek_2)(x)
    
    x = Convolution2D(expand, (1, 3), padding='same' , name=s_id + conv1_3)(x)
    if (batch):
        x = BatchNormalization()(x)
    x = Activation('relu', name=s_id + relu + conv1_3)(x)
    
    x = Convolution2D(expand, (3, 1), padding='same' , name=s_id + conv3_1)(x)
    if (batch):
        x = BatchNormalization()(x)
    x = Activation('relu', name=s_id + relu + conv3_1)(x)
    
    x = Convolution2D(expand2, (1, 1), padding='valid', name=s_id + exp1x1)(x)
    if (batch):
        x = BatchNormalization()(x)
    x = Activation('relu', name=s_id + relu + exp1x1)(x)
    
    x = add([in_p,x], name=s_id + 'add')
    return x


def SqueezeNext150(input_shape=None, classes=1):
    img_input = Input(shape=input_shape)
    #  28 -> 13 
    x = Convolution2D(32, (4, 4), strides=(2, 2), padding='valid', name='conv1')(img_input)
    x = BatchNormalization()(x)
    x = Activation('relu', name='relu_conv1')(x)
    #  13  -> 6
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool1')(x)
    x = Convolution2D(64, (1, 1), strides=(1, 1), padding='valid', name='conv2')(x)
    for i in range(2):
        x = fire_next_module(x, fire_id=i,    squeeze=4, expand=16, expand2=64)
        x = BatchNormalization()(x)
    #  6  -> 2
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool17')(x)
   
    x = Flatten()(x)
    x = Dense(2)(x)
    ip1 = PReLU(name='ip1')(x)
    ip2 = Dense(num_classes)(ip1)
    ip2 = Lambda(lambda x: tf.nn.softmax(x),name='ip2')(ip2)
    
    model = Model(inputs=inputs, outputs=[ip2])
    return model

def centerl_loss_layers(model, num_classes):
    input_target = Input(shape=(1,),name='input_2') # single value ground truth labels as inputs
    centers = Embedding(10,2)(input_target)
    l2_loss = Lambda(lambda x: K.sum(K.square(x[0]-x[1][:,0]),1,keepdims=True),name='l2_loss')([model.get_layer('ip1').output,centers])
    model_centerloss = Model(inputs=[model.input,input_target],outputs=[model.get_layer('ip2').output, l2_loss]) 
    return model_centerloss

def func(a):
     print (a)
     return True

def constrative_center_loss_layers(model,num_classes):
    vec_dim = 2
    emb_inputs = []
    for i in range(num_classes):
        emb_inputs.append(Input(shape=(1,), name='input_{}'.format(str(i+6))))
    center_model_iput = Input(shape=(1,), name='input_2') 
    centers = Embedding(10,2)(center_model_iput)
    centers = Model(center_model_iput, centers)
    centers_arr = []
    for i in range(num_classes):
        centers_arr.append(centers(emb_inputs[i]))
    l2_loss = Lambda(lambda x : K.sum(K.square(x[0]-x[1][:,0]),keepdims=True))([model.get_layer('ip1').output, centers_arr[0]])
    l2_l2_loss = Lambda(lambda x : (1/(num_classes**2 - num_classes))*K.sum([K.sum([K.square(b[:,0] - a[:,0]) for b in centers_arr if a!=b]) for a in centers_arr]))(centers_arr)
    loss = Lambda(lambda x: x[0] + x[1] if func(x[0] + x[1]) else 0)([l2_loss,l2_l2_loss])
    model_centerloss = Model(inputs=[model.input] + emb_inputs, outputs=[model.get_layer('ip2').output,l2_loss]) 
    return model_centerloss


def get_model():
    img_input = Input(shape=(28,28,1))
    #  28 -> 13 
    x = Convolution2D(32, (4, 4), strides=(2, 2), padding='valid', name='conv1')(img_input)
    x = BatchNormalization()(x)
    x = Activation('relu', name='relu_conv1')(x)
    #  13  -> 6
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool1')(x)
    x = Convolution2D(64, (1, 1), strides=(1, 1), padding='valid', name='conv2')(x)
    for i in range(2):
        x = fire_next_module(x, fire_id=i,    squeeze=4, expand=16, expand2=64)
        x = BatchNormalization()(x)
    #  6  -> 2
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool17')(x)
   
    x = Flatten()(x)
    x = Dense(2)(x)
    ip1 = Activation('relu',name='ip1')(x)
    ip2 = Dense(10)(ip1)
    ip2 = Lambda(lambda x: tf.nn.softmax(x),name='ip2')(ip2)
    
    model = Model(inputs=img_input, outputs=[ip2])
    return model


#model = get_model()
#model.compile(loss=["categorical_crossentropy"],
#              loss_weights=[1],
#              optimizer=SGD(lr=0.05),
#              metrics=['accuracy'])
#model.fit(x_train, y_train, batch_size=batch_size, epochs=10, verbose=1)
#model.save('pretrain.hdf5')
#model = load_model('pretrain.hdf5', custom_objects={'tf':tf})
model = get_model()
model = constrative_center_loss_layers(model, 10) 
print (model.summary())
iscenter=True
model.compile(loss=["categorical_crossentropy", lambda y_true,y_pred:y_pred],
              loss_weights=[1,0.0005],
              optimizer=SGD(lr=0.05),
              metrics=['accuracy'])
histories = Histories(iscenter,'constrative-center')

not_y_train_value = [[i for i in range(num_classes) if i!=b] for b in y_train_value]
not_y_train_value = np.array([np.array(b) for b in not_y_train_value])

not_y_test_value = [[i for i in range(num_classes) if i!=b] for b in y_test_value]
not_y_test_value = np.array([np.array(b) for b in not_y_test_value])

model.fit([x_train,y_train_value] +  [not_y_train_value[:,i] for i in range(num_classes-1)], [y_train,np.random.rand(x_train.shape[0],1)], batch_size=batch_size, epochs=epochs, verbose=1, validation_data=([x_test,y_test_value] + [not_y_test_value[:,i] for i in range(num_classes-1)],[y_test,np.random.rand(x_test.shape[0],1)]), callbacks=[histories])


