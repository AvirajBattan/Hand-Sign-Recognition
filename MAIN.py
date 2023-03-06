
#IMPORTING LIBRARY

import cv2
from cvzone.HandTrackingModule import HandDetector
from matplotlib.pyplot import fill
import numpy as np
from math import ceil
from time import time
import tensorflow
from cvzone.ClassificationModule import Classifier


#created the object for dectection
cap=cv2.VideoCapture(0)
detector= HandDetector(maxHands=1,minTrackCon=0.7)


#just a variable for use in further process.
offset=30

#creating classifire for the classification
classifier=Classifier("model\keras_model.h5","model\labels.txt")
labels=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'R', 'V', 'W', 'Y']

winsize=300

counter=0

while True:

    success,img=cap.read()
    outputimg=img.copy()

    hands,img=detector.findHands(img)

    if hands:

        # to get hand here we are using 1 hand only if we want to use other hand as  well we can use hands[1]
        hands=hands[0]
        
        #getting bbox info.
        x,y,w,h = hands["bbox"]
        
        #fixed size matrix.
        imgcrop=img[y-offset:y+h+offset,x-offset:x+w+offset]
        imgwhite=np.ones((winsize,winsize,3),np.uint8)*255
        
        #MAINTIANG RESOLUTION
        ratio_resolution=h/w
        
        if h>=w:
            hspace=0
            hcalulated=300
            wcalculated=ceil(winsize/ratio_resolution)
            wspace=(winsize-wcalculated)//2
        else:
            wspace=0
            wcalculated=300
            hcalulated=ceil(winsize*ratio_resolution)
            hspace=(winsize-hcalulated)//2

        imgresize=cv2.resize(imgcrop,(wcalculated,hcalulated))
        
        imgwhite[hspace:hspace+imgresize.shape[0],wspace:wspace+imgresize.shape[1]]=imgresize
        
        prediction,index=classifier.getPrediction(imgwhite)

        print(prediction,index)
        
        #cv2.imshow("imgwhite",imgwhite)
        #cv2.imshow("hand",imgcrop)
   
        cv2.putText(outputimg,labels[index],(x,y-10),cv2.FONT_HERSHEY_COMPLEX,3,(195,23,255),3)

        cv2.rectangle(outputimg,(x-15,y-15),(x+w+15,y+h+15),(195,23,255),4)
    
        #mytxt=labels[index]
    
        #audio=gTTS(text=mytxt,lang="en",tld="com",slow=False)

        #audio.save("prediction.mp3")

        #os.system("prediction.mp3")
    
    cv2.imshow("image",outputimg)
    
    key=cv2.waitKey(1)

    if key==ord("q"):
        break