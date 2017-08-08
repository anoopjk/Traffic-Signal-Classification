# -*- coding: utf-8 -*-
"""
Created on Sat Jul 15 11:21:42 2017

@author: Anoop
"""

import os
import shutil
import csv

# training confs
BATCH_SIZE = 64
TRAINING_EPOCHS = 200 #max
TRAIN_IMAGES_PER_EPOCH = 16768
VALIDATE_IMAGES_PER_EPOCH = 1856
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
#arrange the images in 3 folders green, red, no
basedir = 'D:/deeplearning_datasets/nexar_trafficlight_app/trainingdata/nexar_traffic_light_train/'
traindir = 'D:/deeplearning_datasets/nexar_trafficlight_app/trainingdata/train/'
validdir = 'D:/deeplearning_datasets/nexar_trafficlight_app/trainingdata/validation/'
nofolder = "no0/"
greenfolder = "green2/"
redfolder = "red1/"
count = 0

#src = os.path.join('C:', 'Documents and Settings', 'user', 'Desktop', 'FilesPy')
#des = os.path.join('C:', 'Documents and Settings', 'user', 'Desktop', 'tryPy', 'Output')
#srcFile = os.path.join('C:', 'Documents and Settings', 'user', 'Desktop', 'tryPy', 'Input')

with open('D:/deeplearning_datasets/nexar_trafficlight_app/trainingdata/nexar_traffic_light_train/labels.csv', 'rb') as csvfile:
    ground_truth = csv.reader(csvfile, delimiter= ',', quotechar = '|')
    for count, row in enumerate(ground_truth):
        #print row
        img_name, label = row
        #print label,img_name
        if count <= TRAIN_IMAGES_PER_EPOCH:
           # print "entered if"
            #print label == "0"
            if label == "0":
                #print "label" , label
                #place the image in No folder
                shutil.copy(basedir + img_name, traindir + nofolder)
            elif label == "1":
                #place the image in Red folder
                shutil.copy(basedir+ img_name, traindir + greenfolder)
            elif label == "2":
                #place the image in Green folder
                shutil.copy(basedir+ img_name, traindir + redfolder)
                
        elif count > TRAIN_IMAGES_PER_EPOCH:                     #and count <= TRAIN_IMAGES_PER_EPOCH + VALIDATE_IMAGES_PER_EPOCH:
             #print "entered elif"
             if label == "0":
                #place the image in No folder
                shutil.copy(basedir+ img_name, validdir + nofolder)
             elif label == "1":
                #place the image in Red folder
                shutil.copy(basedir+ img_name, validdir + greenfolder)
             elif label == "2":
                #place the image in Green folder
                shutil.copy(basedir+ img_name, validdir + redfolder)
                
       
            
        