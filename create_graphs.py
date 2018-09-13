import numpy as np
import skimage
from skimage import data
from matplotlib import pyplot as plt

import numpy as np
import cv2
import pandas as pd 

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
from sklearn import metrics
from sklearn.utils import class_weight


colors = ['aqua', 'cornflowerblue','darkorange', 'red', 'green', 'purple', 'yellow', 'black']
from matplotlib.ticker import AutoMinorLocator
import matplotlib.ticker as ticker
from sklearn.metrics import f1_score
from sklearn.metrics import fbeta_score

def fbeta(y_true, y_pred, threshold_shift=0):
    beta = 0.5

    y_pred = K.clip(y_pred, 0, 1)
   # y_pred_bin = (y_pred + threshold_shift)

    tp = K.sum((y_true * y_pred)) + K.epsilon()
    fp = K.sum((K.clip(y_pred - y_true, 0, 1)))
    fn = K.sum((K.clip(y_true - y_pred, 0, 1)))

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    beta_squared = beta ** 2
    return -(beta_squared + 1) * (precision * recall) / (beta_squared * precision + recall + K.epsilon())

def fbeta_or(y_true, y_pred, threshold_shift=0):
    beta = 0.5

    # just in case of hipster activation at the final layer
    y_pred = np.clip(y_pred, 0, 1)

    # shifting the prediction threshold from .5 if needed
    y_pred_bin = np.round(y_pred + threshold_shift)

    tp = sum((y_true * y_pred_bin))
    fp = sum((np.clip(y_pred_bin - y_true, 0, 1)))
    fn = sum((np.clip(y_true - y_pred, 0, 1)))

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    beta_squared = beta ** 2
    return (beta_squared + 1) * (precision * recall) / (beta_squared * precision + recall + K.epsilon())

def weighted_binary_crossentropy(zero_weight, one_weight):

    def loss(y_true, y_pred):

        # Original binary crossentropy (see losses.py):
        # K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)

        # Calculate the binary crossentropy
        b_ce = K.binary_crossentropy(y_true, y_pred)

        # Apply the weights
        weight_vector = y_true * one_weight + (1. - y_true) * zero_weight
        weighted_b_ce = weight_vector * b_ce

        # Return the mean error
        return K.mean(weighted_b_ce)

    return loss

def custom_acc(y_true, y_pred):
    intersection = K.sum(K.abs(y_true * K.round(y_pred)),axis=-1)
    ones  = K.sum(K.round(y_pred),axis=-1)
    #oness = K.sum(y_true,axis=-1)
    a = intersection + 1
   # b = K.max(ones,oness) + 1
    acc = a/ones
    return acc

def focal_loss_sigmoid(labels,logits,alpha=0.25,gamma=2):
    """
    Computer focal loss for binary classification
    Args:
      labels: A int32 tensor of shape [batch_size].
      logits: A float32 tensor of shape [batch_size].
      alpha: A scalar for focal loss alpha hyper-parameter. If positive samples number
      > negtive samples number, alpha < 0.5 and vice versa.
      gamma: A scalar for focal loss gamma hyper-parameter.
    Returns:
      A tensor of the same shape as `lables`
    """
    y_pred=tf.nn.sigmoid(logits)
    labels=tf.to_float(labels)
    L=-labels*(1-alpha)*((1-y_pred)*gamma)*tf.log(y_pred)-\
      (1-labels)*alpha*(y_pred**gamma)*tf.log(1-y_pred)
    return L

def frange(start, stop, step):
    i = start
    while i < stop:
        yield i
        i += step

def theshold_curve(Y_true, Y_pred, names):
    plt.clf()
    plt.figure()
    fig, ax = plt.subplots()
    Y_t = np.array(Y_true)
    for j in xrange(len(names)):
        Y_true = np.array(Y_t)
        x,y,ysep = [],[],[]
        Y_true_inv = 1 - Y_true
        for thrsh in frange(0,1,0.01):
            Y_tmp = np.array(Y_pred[names[j]][:,1], np.float32)
            Y_inv = np.array(Y_pred[names[j]][:,1], np.float32)
            for i in xrange(len(Y_tmp)):
                Y_tmp[i] = 1 if Y_tmp[i] > thrsh else 0
                Y_inv[i] = 1 - Y_tmp[i]
            c = [a*b for a,b in zip(Y_true, Y_tmp)]
            g = [a*b for a,b in zip(Y_true_inv, Y_inv)]
            acc = float(sum(c)) / sum(Y_true)
            y.append(acc)
            ac = float(sum(g)) / sum(Y_true_inv)
            ysep.append(ac)
            x.append(thrsh)
#        f1score = f1_score(Y_true,Y_tmp)    
        ax.plot(x,y, colors[j],label=names[j])
        ax.plot(x,ysep, colors[j])
    plt.ylim([0.0, 1.0])
    plt.xlim([0.0, 1.0])
#    plt.yticks([thrsh for thrsh in frange(0,1.1,0.05)])
    minor_locator = ticker.LinearLocator(101)
    ax.xaxis.set_major_locator(ticker.LinearLocator(6))
    ax.yaxis.set_major_locator(ticker.LinearLocator(21))
    ax.xaxis.set_minor_locator(ticker.LinearLocator(21))
    ax.yaxis.set_minor_locator(minor_locator)
    ax.grid(True, which='minor', linestyle='--')
    ax.grid(True, which='major', linestyle='-')
    plt.title('Accurecy on hresholds curve')
    plt.xlabel('Thresholds')
    plt.ylabel('Accuracy')
    plt.legend(loc="lower right")
    plt.savefig('threshold.png', dpi = 500)
    plt.show()
    plt.clf()


def precision_recal_(Y_true, Y_pred, names):
    lw = 2
    fig, ax = plt.subplots()

    for i in xrange(len(names)):
        precision, recal, _ = precision_recall_curve(Y_true, Y_pred[names[i]][:,1])
        ap = average_precision_score(Y_true, Y_pred[names[i]][:,1])
        ax.plot(precision,recal,colors[i],lw=lw,label='Precision-Recall of {0} : AP= {1:0.4f}'.format(names[i], ap))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.0])
    plt.xlim([0.0, 1.0])
   # plt.yticks([thrsh for thrsh in frange(0,1.1,0.05)])
    minor_locator = AutoMinorLocator(100)
    ax.xaxis.set_major_locator(ticker.LinearLocator(6))
    ax.xaxis.set_minor_locator(minor_locator)
    ax.yaxis.set_minor_locator(minor_locator)
    ax.grid(True)
    ax.grid(which='minor')
    plt.title('Precision-Recall curves')
    plt.legend(loc="lower right")
    plt.show()
    plt.savefig('ap_curves.png', dpi = 500)
    cv2.waitKey(100)
    plt.clf()

def precision_recal_curve(Y_true, Y_pred, names):
    lw = 2
#    fpr = dict()
#    tpr = dict()
#    roc_auc = dict()
    fig, ax = plt.subplots()
    for i in xrange(len(names)):
        ap = average_precision_score(Y_true, Y_pred[names[i]][:,1])
        precision, recal, _ = precision_recall_curve(Y_true, Y_pred[names[i]][:,1])
        f1score = 2 * (precision[len(precision)/2] * recal[len(recal)/2]) / (precision[len(precision)/2] + recal[len(recal)/2])
        plt.plot(precision,recal,colors[i],lw=lw,label='ROC of {0} (area = {1:0.4f})'.format(names[i], ap, f1score))
#    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    #plt.yticks([thrsh for thrsh in frange(0,1.1,0.05)])
    minor_locator = ticker.LinearLocator(101)
    ax.xaxis.set_major_locator(ticker.LinearLocator(6))
    ax.yaxis.set_major_locator(ticker.LinearLocator(21))
    ax.xaxis.set_minor_locator(ticker.LinearLocator(21))
    ax.yaxis.set_minor_locator(minor_locator)
    ax.grid(True, which='minor', linestyle='--')
    ax.grid(True, which='major', linestyle='-')
    plt.title('Precision-Recall curves')
    plt.legend(loc="lower right")
    plt.savefig('ap_curves.png', dpi = 500)
    plt.show()
    plt.clf() 
    
def roc_curve(Y_true, Y_pred, names):
    lw = 2
#    fpr = dict()
#    tpr = dict()
#    roc_auc = dict()
    for i in xrange(len(names)):
        fpr,tpr, thresholds = metrics.roc_curve(Y_true, Y_pred[names[i]][:,1])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr,tpr,colors[i],lw=lw,label='ROC of {0} (area = {1:0.4f})'.format(names[i], roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.title('ROC curves')
    plt.legend(loc="lower right")
    plt.savefig('roc_curves.png', dpi = 700)
    plt.show()
    plt.clf()
    
def fbeta_curve(Y_true, Y_pred, names):
    plt.clf()
    plt.figure()
    fig, ax = plt.subplots()
    Y_t = np.array(Y_true)
    for j in xrange(len(names)):
        Y_true = np.array(Y_t)
        x,y,ysep = [],[],[]
        Y_true_inv = 1 - Y_true
        for thrsh in frange(0,1,0.01):
            Y_tmp = np.array(Y_pred[names[j]][:,1], np.float32)
            Y_inv = np.array(Y_pred[names[j]][:,1], np.float32)
            for i in xrange(len(Y_tmp)):
                Y_tmp[i] = 1 if Y_tmp[i] > thrsh else 0
                Y_inv[i] = 1 - Y_tmp[i]
            c = [a*b for a,b in zip(Y_true, Y_tmp)]
            g = [a*b for a,b in zip(Y_true_inv, Y_inv)]
            acc = fbeta_or(Y_true, Y_tmp)

            y.append(acc)
            x.append(thrsh)
#        f1score = f1_score(Y_true,Y_tmp)    
        ax.plot(x,y, colors[j],label=names[j])
    plt.ylim([0.0, 1.0])
    plt.xlim([0.0, 1.0])
#    plt.yticks([thrsh for thrsh in frange(0,1.1,0.05)])
    minor_locator = ticker.LinearLocator(101)
    ax.xaxis.set_major_locator(ticker.LinearLocator(6))
    ax.yaxis.set_major_locator(ticker.LinearLocator(21))
    ax.xaxis.set_minor_locator(ticker.LinearLocator(21))
    ax.yaxis.set_minor_locator(minor_locator)
    ax.grid(True, which='minor', linestyle='--')
    ax.grid(True, which='major', linestyle='-')
    plt.title('f1 on Thresholds curve')
    plt.xlabel('Thresholds')
    plt.ylabel('f1_score')
    plt.legend(loc="lower right")
    plt.savefig('f1_score.png', dpi = 500)
    plt.show()
    plt.clf()



def create_graphs(Y_true, Y_pred, names):
 #   theshold_curve(Y_true, Y_pred, names)
 #   precision_recal_curve(Y_true, Y_pred, names)
 #   roc_curve(Y_true, Y_pred, names)
   fbeta_curve(Y_true, Y_pred, names)



def get_val():
    dframe_path = '/home/dmitry.kamarouski/work/abandoned/val_data.csv'
    data = pd.read_csv(dframe_path, header=None, sep = ' ')
    X = []
    Y = []
    for i in range(len(data[0])):
        img = cv2.resize(cv2.imread('/home/dmitry.kamarouski/work/abandoned/indexed/abndom_folders/.'  + data[0][i]),(224, 224))
        X.append(img)
        Y.append(data[1][i])
    X = np.array(X, np.float32) / 255
    return X, Y

def visualize(feat, labels, epoch):

    plt.ion()
    c = ['#ff0000', '#ffff00']
    plt.clf()
    for i in range(2):
        plt.plot(feat[:, 0], feat[:, 1], '.', c=c[i])
    plt.legend(['0', '1'], loc = 'upper right')
    XMax = np.max(feat[:,0]) 
    XMin = np.min(feat[:,1])
    YMax = np.max(feat[:,0])
    YMin = np.min(feat[:,1])

    plt.xlim(xmin=XMin,xmax=XMax)
    plt.ylim(ymin=YMin,ymax=YMax)
    plt.text(XMin,YMax,"epoch=%d" % epoch)
    plt.savefig('./epoch=%d.jpg' % epoch)
    plt.draw()

#weights = np.ones((2,))
#weights[0] = 0.15
#weights[1] = 0.85    
#loss = weighted_binary_crossentropy(weights[0],weights[1]),
#WEIGHTS_PATH = '/home/dmitry.kamarouski/work/rd/abandoned/weights/'
#MODEL_PATH   = '/home/dmitry.kamarouski/work/rd/abandoned/models/'
models_names = ['3d_constractive-center']
X,Y = get_val()
Y_true = {0:0}
for name in models_names :
    #model = load_model(MODEL_PATH   + 'batch_train_{}.hdf5'.format(name))
    #model.load_weights(WEIGHTS_PATH + 'batch_train_{}.hdf5'.format(name))
    #model = load_model(MODEL_PATH   + 'fscore_{}.hdf5'.format(name), custom_objects={'focal_loss_sigmoid':focal_loss_sigmoid, 'fbeta_or':fbeta_or})
    #model.load_weights(MODEL_PATH   + 'fscore_{}.hdf5'.format(name))
    if name == '3d_constractive-center':
        model = load_model('konvert_to_tf/constrative-center_loss_3_Xception_089.hdf5',custom_objects={'fbeta_or':fbeta_or})
    elif name == 'DenseNet121':
        model = load_model('models/constrative-center_loss_Densenet121.hdf5',custom_objects={'fbeta_or':fbeta_or})
    elif name == 'DenseNet201':
        model = load_model('models/constrative-center_loss_InceResV1.hdf5',custom_objects={'fbeta_or':fbeta_or})
    Y_1 = model.predict(X)
    Y_true[name] = Y_1
create_graphs(Y, Y_true, models_names )
