import cv2
import time
import os
import sys
from pathlib import Path

root_dir = str(Path(__file__).parent.parent)
sys.path.append(root_dir)

from Modules.Hand_Tracking_Module.hand_tracking_module import HandDetector

wCam, hCam = 640, 480
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

folderPath = "FingerImages"
overlayList = []
for imPath in os.listdir(folderPath):
    image = cv2.imread(f"{folderPath}/{imPath}")
    image = cv2.resize(image, (200, 200))  # Resize all overlays to 200x200
    overlayList.append(image)

detector = HandDetector(min_detection_confidence=0.7)
pTime = 0
tipIds = [4,8,12,16,20]
totalFingers=0

while True:
    success, img = cap.read()
    if not success:
        break
        
    img = detector.find_hands(img)
    lmList = detector.find_position(img, draw=False)
    #print(lmList)
    
    if len(lmList)!=0:
        fingers = []
        
        # Thumb
        if lmList[tipIds[0]][1] > lmList[tipIds[0]-1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
                
        # Other 4 fingers
        for id in range(1,5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
                
        #print(fingers)
        totalFingers = fingers.count(1)
        print(totalFingers)
    
        h, w = 200, 200
        if len(overlayList) > 0:
            img[0:h, 0:w] = overlayList[totalFingers-1]
            
        cv2.rectangle(img, (20,225), (170,425), (0,255,0), cv2.FILLED)
        cv2.putText(img, str(totalFingers), (45,375), cv2.FONT_HERSHEY_PLAIN, 10, (255,0,0), 25)
    
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f"FPS: {int(fps)}", (10, 30), 
                cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
    
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()