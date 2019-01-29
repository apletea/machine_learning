import cv2
import numpy as np
import random
import skimage
from skimage import data
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
import os
import glob
import tensorflow as tf
import keras


from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras.applications import *
from keras.regularizers import *
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from keras import backend as K
from keras.models import load_model
from keras.utils import plot_model
from keras.datasets import mnist




class GAN():
    def __init__(self):
        self.latent_dim = 100
        self.img_rows = 28
        self.img_cols = 28
        self.img_channels = 1
        self.batch_size = 32
        self.num_classes = 10

        self.img_shape = (self.img_rows, self.img_cols, self.img_channels)
        self.optimizer = Adam(0.0002, 0.5)

        self.discriminator = self.build_dicriminator()

        self.discriminator.compile(loss='binary_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])


        self.generator = self.build_generator()

        self.latent_input = Input(shape=(self.latent_dim,))
        self.img = self.generator(self.latent_input)

        self.discriminator.traianable = False

        self.validity = self.discriminator(self.img)

        self.pipeLine = Model(input=self.latent_input, output=self.validity)
        self.pipeLine.compile(loss='binary_crossentropy', optimizer=self.optimizer)

    def build_generator(self):
        model = Sequential()

        model.add(Dense(256, input_shape=(self.latent_dim,)))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(256, input_shape=(self.latent_dim,)))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.img_shape)))
        model.add(Activation('tanh'))
        model.add(Reshape(self.img_shape))

        model.summary()
        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise,img)

    def build_dicriminator(self):
        model = Sequential()

        model.add(Flatten(input_shape=self.img_shape))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))

        model.summary()
        img = Input(shape=(self.img_shape))
        validity = model(img)

        return Model(img, validity)

    def train(self, num_iterations):
        (X_train, _), (_,_) = mnist.load_data()

        X_train = X_train / 127.5 - 1
        X_train = np.expand_dims(X_train, axis=3)

        valid = np.ones((self.batch_size, 1))
        adversarial = np.zeros((self.batch_size, 1))
        for iterations in range(num_iterations):

            idx = np.random.randint(0,X_train.shape[0], self.batch_size)
            imgs = X_train[idx]

            noise = np.random.normal(0,1, (self.batch_size, self.latent_dim))

            gen_img = self.generator.predict(noise)

            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_img, adversarial)

            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake) 

            noise = np.random.normal(0,1, (self.batch_size, self.latent_dim))
            
            g_loss = self.pipeLine.train_on_batch(noise, valid)

            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (iterations, d_loss[0], 100*d_loss[1], g_loss))


            if (iterations % 100 == 0):
                self.samples_images(iterations)


    def samples_images(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images/%d.png" % epoch)
        plt.close()

gan = GAN()
gan.train(num_iterations=10000)
