import cv2
import time
import numpy as np
import os
import sys
from pathlib import Path

root_dir = str(Path(__file__).parent.parent)
sys.path.append(root_dir)

from Modules.Hand_Tracking_Module.hand_tracking_module import HandDetector

folderPath = "Header"
myList = os.listdir(folderPath)

overlayList = []
lmList = []
xp=0
yp=0

for imPath in myList:
    image = cv2.imread(f"{folderPath}/{imPath}")
    overlayList.append(image)
    
header = overlayList[0]
drawColor = (114,255,193)
brushThickness=15
eraserThickness=50
imgCanvas = np.zeros((720,1280,3),np.uint8)

cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

detector = HandDetector(min_detection_confidence=0.85)

while True:
    success, img = cap.read()
    img = cv2.flip(img,1)
    
    img = detector.find_hands(img)
    lmList,_ = detector.find_position(img, draw=False)
    
    if len(lmList)!=0:
        #print(lmList)
        
        # tip of index and middle fingers
        x1,y1 = lmList[8][1:]
        x2,y2 = lmList[12][1:]
        
        print("position of index finger",x1)
        
        fingers = detector.fingers_up()
        #print(fingers)
        
        if fingers[1] and fingers[2]:
            print("Selection Mode")
            xp,yp=0,0
            if y1<125:
                if 250<x1<450:
                    header = overlayList[0]
                    drawColor = (114,255,193)
                elif 540<x1<750:
                    header = overlayList[1]
                    drawColor = (235,23,94)
                elif 800<x1<950:
                    header = overlayList[2]
                    drawColor = (49,49,255)
                elif 1050<x1<1200:
                    header = overlayList[3]
                    drawColor = (0,0,0)
                    
            cv2.rectangle(img,(x1,y1-25),(x2,y2+25),drawColor,cv2.FILLED)
            
        if fingers[1] and fingers[2]==False:
            cv2.circle(img,(x1,y1),15,drawColor,cv2.FILLED)
            print("Drawing Mode")
            if xp==0 and yp==0:
                xp,yp = x1,y1
            
            if drawColor==(0,0,0):
                cv2.line(img,(xp,yp),(x1,y1),drawColor,eraserThickness)
                cv2.line(imgCanvas,(xp,yp),(x1,y1),drawColor,eraserThickness)
            else: 
                cv2.line(img,(xp,yp),(x1,y1),drawColor,brushThickness)
                cv2.line(imgCanvas,(xp,yp),(x1,y1),drawColor,brushThickness)
                
            xp, yp = x1,y1
    
    imgGray = cv2.cvtColor(imgCanvas,cv2.COLOR_BGR2GRAY)
    _,imgInv = cv2.threshold(imgGray,50,255,cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv,cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img,imgInv)
    img = cv2.bitwise_or(img,imgCanvas)
    
            
    header = cv2.resize(header, (1280, 125))
    img[0:125,0:1280] = header
    img = cv2.addWeighted(img,0.5,imgCanvas,0.5,0)
    
    cv2.imshow("Image",img)
    # cv2.imshow("Image Canvas",imgCanvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()