import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import math
import time
import os
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
model = tf.keras.models.load_model('sign_language_model.h5')

# Define a list of class labels corresponding to your sign gestures
class_labels = ['A','B',"C"]#this list based on your dataset
# class_labels = []#this list based on your dataset
# file = open('class_labels.txt','r')
# for i in file.read():
#     if(i != '\n'):
#         class_labels.append(i)
# class_labels.append(list(file.read()))
offset = 50
imgSize = 300
folder = 'data_me/E'
i = 0
while True:
    success,img = cap.read()
    hands,img = detector.findHands(img)
    if hands:
        try:
            hand = hands[0]
            x,y,w,h = hand['bbox']
            imgWhite = np.ones((imgSize,imgSize,3),np.uint8)*255
            imgcrop = img[y-offset:y+h+offset,x-offset:x+offset+w]
            shape_imagecrop = imgcrop.shape
            imgWhite[:shape_imagecrop[0],:shape_imagecrop[1]] = imgcrop
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            aspect_ratio = h/w
            if(aspect_ratio>1):   
                k = imgSize/h
                wcal = math.ceil(k*w)
                imgResize = cv2.resize(imgcrop,(wcal,imgSize))
                imgResizeShape = imgResize.shape
                wGap = math.ceil((imgSize-wcal)/2)
                imgWhite[:,wGap:wGap+wcal] = imgResize
            else:
                k = imgSize/w
                hcal = math.ceil(k*h)
                imgResize = cv2.resize(imgcrop,(imgSize,hcal))
                imgResizeShape = imgResize.shape
                hGap = math.ceil((imgSize-hcal)/2)
                imgWhite[hGap:hGap+hcal,:] = imgResize
            input_data = np.expand_dims(imgWhite, axis=-1)  # Add channel dimension
            input_data = np.expand_dims(input_data, axis=0)  # Add batch dimension
            prediction = model.predict(input_data)
            predicted_class = np.argmax(prediction)
            predicted_label = class_labels[predicted_class]
            cv2.putText(img, f'Predicted: {predicted_label}', (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('white img', imgWhite)

        except Exception as e:
            print(e)

    cv2.imshow('Sign Language Recognition', img)

    key = cv2.waitKey(1)
    if key == ord('s'):
        i+=1
        print(folder)
        cv2.imwrite(f'{folder}/image_{time.time()}.jpg',imgWhite)
        print(i)
    elif(key!=-1):
        print(chr(key))
        folder = f'data_me/{chr(key)}'
        print(folder)
    