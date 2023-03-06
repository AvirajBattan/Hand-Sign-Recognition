import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
from math import ceil
from time import time

#created the object for dectection
cap=cv2.VideoCapture(0)
detector= HandDetector(maxHands=1,minTrackCon=0.7)

#just a variable for use in further process.
offset=30

winsize=300

counter=0

while True:

    success,img=cap.read()
    hands,img=detector.findHands(img)

    if hands:

        # to get hand here we are using 1 hand only if we want to use other hand as  well we can use hands[1]
        hands=hands[0]
        
        #getting bbox info.
        x,y,w,h = hands["bbox"]
        
        #fixed size matrix.
        imgcrop=img[y-offset:y+h+offset,x-offset:x+w+offset]
        imgwhite=np.ones((winsize,winsize,3),np.uint8)*255
        
        
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
        
            

        cv2.imshow("imgwhite",imgwhite)
        cv2.imshow("hand",imgcrop)
    
    cv2.imshow("image",img)
    
    
    key=cv2.waitKey(1)

    if key==ord("q"):
        break

    if key==ord("s"):
        counter+=1
        cv2.imwrite(f'data\V\img_{time()}.jpeg',imgwhite)
        print(counter)