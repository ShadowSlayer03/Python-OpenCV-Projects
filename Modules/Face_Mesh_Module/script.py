import cv2
import mediapipe as mp
import time
from mediapipe.python.solutions.face_mesh import FACEMESH_TESSELATION

class FaceMeshDetector():
    def __init__(self, staticMode=False, maxFaces=2, refineLandmarks=False, minDetectionConf=0.5, minTrackConf=0.5):
        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.refineLandmarks = refineLandmarks
        self.minDetectionConf = minDetectionConf
        self.minTrackConf = minTrackConf

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(
            static_image_mode=self.staticMode,
            max_num_faces=self.maxFaces,
            refine_landmarks=self.refineLandmarks,
            min_detection_confidence=self.minDetectionConf,
            min_tracking_confidence=self.minTrackConf
        )

        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=2)
        
    def find_faces(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(imgRGB)
        faces = []
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms, FACEMESH_TESSELATION, self.drawSpec, self.drawSpec)  # ✅ Fixed here
                
                face = []
                for id, lm in enumerate(faceLms.landmark):
                    ih, iw, ic = img.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)  # ✅ Fixed multiplication issue
                    print(id, x, y) 
                    face.append([x, y]) 
                faces.append(face)
        return img, faces
                
 
def main():
    
    cap = cv2.VideoCapture("videos/beauty.mp4")
    pTime=0 
    detector = FaceMeshDetector()

    while True:
        success,img = cap.read()
        img,faces = detector.find_faces(img)
        
        if len(faces)!=0:
            print(f"Faces:",faces)

        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime=cTime
        cv2.putText(img, f'FPS: {int(fps)}', (20,70),cv2.FONT_HERSHEY_PLAIN,3,(0,255,0),2)
        
        resized_img =cv2.resize(img, (800, 500))
        cv2.imshow("Image",resized_img)
        cv2.waitKey(1)
    
    
if __name__=="__main__":
    main()