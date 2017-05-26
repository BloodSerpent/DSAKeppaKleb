'''
Created on May 26, 2017

@author: miko
'''
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
import os
from scipy import misc
import theano

import numpy as np
import pandas as pd


#truck = 1, mobil 0
image = []
label = []

# train_datagen = ImageDataGenerator()
# train_generator = train_datagen.flow_from_directory("/home/miko/workspace/KeppaKleb/truck",
#         target_size=(32, 32),
#         class_mode='binary')

for filename in os.listdir("/home/miko/workspace/KeppaKleb/truck"):
    i = misc.imread("/home/miko/workspace/KeppaKleb/truck/"+filename)
    label.append(1)
    image.append(np.array(i))


for filename in os.listdir("/home/miko/workspace/KeppaKleb/automobile"):
    i = misc.imread("/home/miko/workspace/KeppaKleb/automobile/"+filename)
    label.append(0)
    image.append(np.array(i))
# image = np.array(image)
print np.shape(image)
# data = pd.DataFrame({"image":image, "label":label})
image = np.asarray(image)
label = np_utils.to_categorical(label,2)


# print data
model = Sequential()
model.add(Convolution2D(input_shape=(32,32,3), filters=6,kernel_size=(5,5),activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Convolution2D(filters=16, kernel_size=(5,5), activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Convolution2D(filters=120, kernel_size=(5,5), activation='relu'))
model.add(Dense(84,activation='tanh'))
model.add(Flatten())
model.add(Dense(2,activation='softplus'))

model.compile(loss='binary_crossentropy', optimizer='adamax', metrics=["accuracy"])
model.fit(image,label,nb_epoch=15)
print model.summary()
