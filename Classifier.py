'''
Created on May 26, 2017

@author: miko
'''
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from sklearn.cross_validation import train_test_split
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.layers.advanced_activations import LeakyReLU
import os

from scipy import misc
import theano

import numpy as np
import pandas as pd

class Classifier:
    def __init__(self,x_train,y_train):
        np.random.seed(10)
        self.x_train = x_train
        self.y_train = y_train
        
        self.model = Sequential()
        self.model.add(Convolution2D(input_shape=(32,32,3), filters=6,kernel_size=(5,5)))
        self.model.add(LeakyReLU(alpha=0.3))
        self.model.add(MaxPooling2D(2,2))
        # model.add(Dropout(0.1))
        self.model.add(Convolution2D(filters=16, kernel_size=(5,5)))
        self.model.add(LeakyReLU(alpha=0.3))
        self.model.add(MaxPooling2D(2,2))
        # model.add(Dropout(0.15))
        self.model.add(Convolution2D(filters=120, kernel_size=(5,5)))
        self.model.add(LeakyReLU(alpha=0.3))
        # model.add(Dropout(0.2))
        self.model.add(Dense(84,activation='tanh'))
        # model.add(Dropout(0.25))
        self.model.add(Flatten())
        self.model.add(Dense(2,activation='sigmoid'))
        self.model.compile(loss='mean_squared_logarithmic_error', optimizer='adamax', metrics=["accuracy"])
    
    def fit(self,batch_size,epoch):
        datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True)
        # compute quantities required for featurewise normalization
        # (std, mean, and principal components if ZCA whitening is applied)
        datagen.fit(self.x_train)
        
        # fits the model on batches with real-time data augmentation:
        self.model.fit_generator(datagen.flow(self.x_train, self.y_train, batch_size=batch_size),steps_per_epoch=len(self.x_train) / batch_size, epochs=epoch)
        
#truck = 1, mobil 0
# image = []
# label = []
# np.random.seed(10)
# # train_datagen = ImageDataGenerator()
# # train_generator = train_datagen.flow_from_directory("/home/miko/workspace/KeppaKleb/truck",
# #         target_size=(32, 32),
# #         class_mode='binary')
# 
# for filename in os.listdir("/home/miko/workspace/KeppaKleb/truck"):
#     i = misc.imread("/home/miko/workspace/KeppaKleb/truck/"+filename)
#     label.append(1)
#     image.append(np.array(i))
# 
# 
# for filename in os.listdir("/home/miko/workspace/KeppaKleb/automobile"):
#     i = misc.imread("/home/miko/workspace/KeppaKleb/automobile/"+filename)
#     label.append(0)
#     image.append(np.array(i))
# # image = np.array(image)
# print np.shape(image)
# # data = pd.DataFrame({"image":image, "label":label})
# # image = np.asarray(image)
# # label = np_utils.to_categorical(label,2)
# (trainData, testData, trainLabel, testLabel) = train_test_split(image, label, test_size=0.1)
# trainData = np.asarray(trainData)
# testData = np.asarray(testData)
# 
# trainLabel = np_utils.to_categorical(trainLabel,2)
# testLabel = np_utils.to_categorical(testLabel,2)
# 
# 
# 
# 
# # print data
# model = Sequential()
# model.add(Convolution2D(input_shape=(32,32,3), filters=6,kernel_size=(5,5)))
# model.add(LeakyReLU(alpha=0.3))
# model.add(MaxPooling2D(2,2))
# # model.add(Dropout(0.1))
# model.add(Convolution2D(filters=16, kernel_size=(5,5)))
# model.add(LeakyReLU(alpha=0.3))
# model.add(MaxPooling2D(2,2))
# # model.add(Dropout(0.15))
# model.add(Convolution2D(filters=120, kernel_size=(5,5)))
# model.add(LeakyReLU(alpha=0.3))
# # model.add(Dropout(0.3))
# model.add(Dense(84,activation='tanh'))
# # model.add(Dropout(0.4))
# model.add(Flatten())
# model.add(Dense(2,activation='sigmoid'))
# 
# model.compile(loss='mean_squared_logarithmic_error', optimizer='adamax', metrics=["accuracy"])
# # model.fit(image,label,nb_epoch=15)
# 
# datagen = ImageDataGenerator(
#     featurewise_center=True,
#     featurewise_std_normalization=True,
#     rotation_range=20,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     horizontal_flip=True)
# 
# # compute quantities required for featurewise normalization
# # (std, mean, and principal components if ZCA whitening is applied)
# datagen.fit(image)
# # fits the model on batches with real-time data augmentation:
# model.fit_generator(datagen.flow(trainData, trainLabel, batch_size=16),steps_per_epoch=len(image) / 16, epochs=200)
# model.save_weights('LeNet5_LeakyRELU.h5')
# (loss, accuracy) = model.evaluate(testData, testLabel, batch_size=16)
# print("{:.2f}%".format(accuracy*100))
