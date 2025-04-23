import cv2
import numpy as np
import time
import sys
from pathlib import Path
import math
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

root_dir = str(Path(__file__).parent.parent)
sys.path.append(root_dir)

from Modules.Hand_Tracking_Module.hand_tracking_module import HandDetector

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)
# volume.GetMute()
# volume.GetMasterVolumeLevel()
volRange = volume.GetVolumeRange()

minVol = volRange[0]
maxVol = volRange[1]
vol=0
volBar=400
volPer=0

wCam, hCam = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

pTime = 0

detector = HandDetector()

while True:
    success, img = cap.read()
    img = detector.find_hands(img)
    lmList = detector.find_position(img,draw=False)
    if len(lmList)!=0:
        print(lmList[4],lmList[8])
        
        x1,y1 = lmList[4][1],lmList[4][2]
        x2,y2 = lmList[8][1],lmList[8][2]
        cx,cy = (x1+x2)//2, (y1+y2)//2
        
        cv2.circle(img,(x1,y1),10,(255,0,255),cv2.FILLED)
        cv2.circle(img,(x2,y2),10,(255,0,255),cv2.FILLED)
        cv2.line(img, (x1,y1), (x2,y2), (255,0,255),3)
        cv2.circle(img,(cx,cy),15,(255,0,255),cv2.FILLED)
        
        length = math.hypot(x2-x1,y2-y1)
        
        # Mapping Hand Range to Volume Range
        vol = np.interp(length,[50,300],[minVol,maxVol])
        volBar = np.interp(length,[50,300],[400,150])
        volPer = np.interp(length,[50,300],[0,100])
        
        print("Volume",vol)
        volume.SetMasterVolumeLevel(vol, None)
        
        if length<50:
            cv2.circle(img, (cx,cy), 10, (0,255,0), cv2.FILLED)
            
    cv2.rectangle(img,(50,150),(85,400),(255,0,0),3)
    cv2.rectangle(img,(50,int(volBar)),(85,400),(255,0,0),cv2.FILLED)  
    cv2.putText(img, f"Volume - {int(volPer)}%", (40, 130), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
    
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    
    cv2.putText(img, f"FPS {int(fps)}", (40, 50), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 1)
    
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()