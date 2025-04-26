import cv2
import time
import numpy as np
import sys
from pathlib import Path

root_dir = str(Path(__file__).parent.parent)
sys.path.append(root_dir)

from Modules.Pose_Estimation_Module.pose_estimation_module import PoseDetector

cap = cv2.VideoCapture("videos/bicep_curls.mp4")

detector = PoseDetector()
count=0
dir=0
pTime=0

while True:
    success, img = cap.read()
    img = cv2.resize(img,(1280,720))
    
    # img = cv2.imread("videos/test1.png")
    img = detector.find_pose(img)
    
    lmList = detector.find_position(img,False)
    if len(lmList)!=0:
        # Right Arm
        # detector.find_angle(img,12,14,16)
        # Left Arm
        angle= detector.find_angle(img,11,13,15)
        per= np.interp(angle,(210,310),(0,100))
        bar= np.interp(angle,(220,310),(650,100))
        
        print(angle,per)
        
        color = (255,0,255)
        if per==100:
            color = (0,255,0)
            if dir==0:
                count+=0.5
                dir=1
        if per==0:
            color = (0,255,0)
            if dir==1:
                count+=0.5
                dir=0
        
        cv2.rectangle(img,(1100,100),(1175,650),color,3)
        cv2.rectangle(img,(1100,int(bar)),(1175,650),color,cv2.FILLED)
        cv2.putText(img,f"{int(per)}%",(1100,75),cv2.FONT_HERSHEY_PLAIN,4,color,4)
                
        cv2.rectangle(img,(0,450),(250,720),(0,255,0),cv2.FILLED)
        cv2.putText(img,f"{int(count)}",(45,670),cv2.FONT_HERSHEY_PLAIN,15,(255,0,0),25)
            
    
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    
    cv2.putText(img, f"FPS: {int(fps)}", (10, 30), 
                cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
    
    cv2.imshow("Image",img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
    
    