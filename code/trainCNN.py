# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 17:27:45 2017

@author: Anoop
"""


import os
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
#from keras.optimizers import Adam
from keras import backend as K

TRAIN_IMAGES_PER_EPOCH = 16769
VALIDATE_IMAGES_PER_EPOCH = 1890
NUM_CLASSES = 3


# dimensions of our images.
img_width, img_height = 224, 224
os.chdir('D:/deeplearning_datasets/nexar_trafficlight_app/')
train_data_dir = 'trainingdata/train/'
validation_data_dir = 'trainingdata/validation/'
nb_train_samples = TRAIN_IMAGES_PER_EPOCH
nb_validation_samples = VALIDATE_IMAGES_PER_EPOCH
weights_path = 'save_model/weights/fourth_try.h5'    


epochs = 20
batch_size = 64

from keras.backend.tensorflow_backend import set_session
import tensorflow as tf

#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
gpu_options = tf.GPUOptions (per_process_gpu_memory_fraction=0.6) 
set_session(tf.Session(config=tf.ConfigProto (gpu_options=gpu_options)))


if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)
    
#define the model
#model = Sequential([
#    Convolution2D(16,3,3, border_mode= 'same', subsample=(2,2), input_shape=input_shape,   activation='relu'),
#    MaxPooling2D(pool_size= (3,3)),
#    Dropout(0.2),
#    
#    Convolution2D(32, 3,3, border_mode= 'same', activation= 'relu'),
#    MaxPooling2D(pool_size=(3,3)),
#    Dropout(0.2),
#
#    Convolution2D(64,3,3, border_mode ='same', activation= 'relu'),
#    MaxPooling2D(pool_size=(2,2)),
#    Dropout(0.2),
#    
#    Flatten(),
#    Dense(128, activation= 'tanh'),
#    Dropout(0.3),
#    Dense(NUM_CLASSES, activation='softmax'),
#
#])
#model.summary()
    # we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
#model.compile(optimizer=Adam(lr=0.0003), loss='categorical_crossentropy', metrics=['accuracy'] )

model = Sequential()
model.add(Conv2D(16, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(0.2))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(NUM_CLASSES))
model.add(Activation('sigmoid'))
adam = optimizers.Adam(lr=0.003)
model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])

model.summary()

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')
    
    
model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)
    


model.save_weights(weights_path)


