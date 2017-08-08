# -*- coding: utf-8 -*-
"""
Created on Sat Jul 15 16:03:05 2017

@author: Anoop
"""

#testing the cnn
from keras.preprocessing import image

#from keras.optimizers import Adam
from keras import backend as K
import os
import numpy as np
import cv2
from model import get_model_6layers_, get_model_5layers_


NUM_CLASSES = 3

os.chdir('D:/deeplearning_datasets/nexar_trafficlight_app/')
weights_path = 'save_model/weights/third_try.h5'         
img_width, img_height = 224, 224 # this is the model image size

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)
    
model = get_model_5layers_()
              
model.load_weights(weights_path)

img_filepath = 'testingdata/Lara_UrbanSeq1_JPG/Lara3D_UrbanSeq1_JPG/'
imgfiles = os.listdir(img_filepath)
for jpgfile in imgfiles:
    if not(jpgfile.endswith(".jpg")):
            imgfiles.remove(jpgfile)

# Write some Text
font = cv2.FONT_HERSHEY_SIMPLEX

    
cv2.namedWindow("dashcam image",  cv2.WINDOW_NORMAL)

for img_name in imgfiles:
    
    
    img_orig = cv2.imread(os.path.join(img_filepath,img_name))
    img_orig = cv2.resize(img_orig, (img_width, img_height), interpolation = cv2.INTER_AREA)
    img = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
    img= img.astype('float64')
    img = np.transpose(img, (2, 1, 0))
    img= np.expand_dims(img, axis=0)

    #img = image.load_img(img_path, target_size=(img_width, img_height))
    #x = image.img_to_array(img)
    #x = np.expand_dims(x, axis=0)
    #x = preprocess_input(x)
    pred = model.predict(img)
    maxindex = np.argmax(pred)
    if (maxindex== 0):
        # no signal
        output = "NONE"
    elif (maxindex == 1):
        #red signal
        output = "RED"
        
    elif(maxindex == 2):
        #green signal
        output = "GREEN"
    
    cv2.putText(img_orig,output,(10,30), font, 1.0, (0,0,255), 3)
    cv2.imshow("dashcam image", img_orig)
    cv2.waitKey(10)
    print output
    print pred

    
cv2.destroyAllWindows()


#green
#green
#no
#no
#red
#red
#red
#red
