'''
Created on May 27, 2017

@author: miko
'''
import cv2
import keras
from keras.models import load_model, Sequential

from BackgroundSubstract import BackgroundSubstractClass
import numpy as np
from Classifier import Classifier

class MyObjects:
    def __init__(self,image,class_type,upleft_location,bottomright_location):
        self.image = image
        self.class_type = class_type
        self.upleft_location = upleft_location
        self.bottomright_location = bottomright_location


cap = cv2.VideoCapture('/home/miko/workspace/VideoTracking/skenario_stop.mp4')
font = cv2.FONT_HERSHEY_SIMPLEX
models = Classifier(None,None)
models.model.load_weights("LeNet5_LeakyRELU.h5")
newObj = BackgroundSubstractClass(int(cap.get(4)),int(cap.get(3)), 0.01)
count = 0
while (cap.isOpened):
    ret ,frame = cap.read()

    if ret:
        model = newObj.updateModel(frame)
        count+=1
        image = np.abs(frame-model).astype(np.uint8)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        image = cv2.GaussianBlur(image, (7, 7), 0)
        image[image >50] = 255
        image[image <=50] = 0
        image = cv2.erode(image,(3,3),iterations = 1)
        image = cv2.dilate(image,None,iterations = 2)
        param = cv2.SimpleBlobDetector_Params()
        param.filterByInertia = False
        param.filterByConvexity = False
        param.filterByColor = True
        param.blobColor = 255
        param.minArea = 700.0
        detector = cv2.SimpleBlobDetector_create(param)
        keypoint = detector.detect(image)
        objects = []
        if count > 400:
            for k in keypoint:
                x,y = k.pt
                x = round(x)
                y = round(y)
                d = k.size
                """
                    This method suffer from empty slicing (m == empty array [])
                """
                c = round(d/2)
                x1 = x - c
                y1 = y - c
                x2 = x + c
                y2 = y + c
                upleft_y_x = (int(y1),int(x1))
                bottom_right_y_x = (int(y2),int(x2))
                a = []
                image_object = frame[int(y1):int(y2),int(x1):int(x2)]
                image_object = cv2.resize(image_object,(32,32))
                a.append(image_object)
                cv2.imshow("object",image_object)
                ob_class = models.model.predict(np.array(a))
                if ob_class[0][0] >= ob_class[0][1]:
                    ob_class = "big ass car"
                else:
                    ob_class = "car"
                ob = MyObjects(image_object,ob_class,upleft_y_x,bottom_right_y_x)
                objects.append(ob)
            
            for ob in objects:
                cv2.rectangle(frame,(ob.upleft_location[1],ob.upleft_location[0]),(ob.bottomright_location[1],ob.bottomright_location[0]),(0,0,255),1)
                cv2.putText(frame,ob.class_type,(ob.upleft_location[1]+2,ob.upleft_location[0]+2),font,0.5,(255,0,0),2)
#         imagedraw = cv2.drawKeypoints(image,keypoint,np.array([]),(0,0,255),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#         cv2.imshow('anjas',imagedraw)
#         cv2.imshow('njz',frame)
        cv2.imshow("result",frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break