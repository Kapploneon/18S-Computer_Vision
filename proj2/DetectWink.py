import numpy as np
import cv2
import os
from os import listdir
from os.path import isfile, join
import sys

def detectWink(frame, location, ROI, cascade, cascadeglasses):
    eyes = cascade.detectMultiScale(ROI, 1.1, 4, 0|cv2.CASCADE_SCALE_IMAGE, (5, 10)) 

    if len(eyes)==0:
        eyes = cascadeglasses.detectMultiScale(ROI, 1.1, 1, 0|cv2.CASCADE_SCALE_IMAGE, (5, 10)) 
    if len(eyes)>=2:
        eyes = cascadeglasses.detectMultiScale(ROI, 1.1, 4, 0|cv2.CASCADE_SCALE_IMAGE, (5, 10)) 

    for e in eyes:
        e[0] += location[0]
        e[1] += location[1]
        x, y, w, h = e[0], e[1], e[2], e[3]        
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0, 0, 255), 2)
    return len(eyes) == 1    # number of eyes is one

def detect(frame, faceCascade, faceCascadeDefault, eyesCascade, eyesCascadeGlasses):

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 

    # possible frame pre-processing:
    # gray_frame = cv2.medianBlur(frame, 5)
    # gray_frame = cv2.equalizeHist(gray_frame)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) #convert it to hsv
    h, s, v = cv2.split(hsv)
    v += 255
    final_hsv = cv2.merge((h, s, v))
    final_img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    gray_frame = cv2.cvtColor(final_img, cv2.COLOR_BGR2GRAY) 

    faces = faceCascade.detectMultiScale(gray_frame, 1.15, 5, 0|cv2.CASCADE_SCALE_IMAGE, (30,30))

    if len(faces)==0:
        faces = faceCascadeDefault.detectMultiScale(gray_frame, 1.1, 2,  0|cv2.CASCADE_SCALE_IMAGE, (30,30)) 
   
    detected = 0

    for f in faces:
        x, y, w, h = f[0], f[1], f[2], f[3]
        
        # faceROI = gray_frame[y:y+h, x:x+w]
        upperROI = gray_frame[y:(y+int(0.6*h)), x:x+w]

        if detectWink(frame, (x, y), upperROI, eyesCascade, eyesCascadeGlasses):
            detected += 1
            cv2.rectangle(frame, (x,y), (x+w,y+h), (255, 0, 0), 2)
        else:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0, 255, 0), 2)
    return detected

def run_on_folder(cascade1, cascade2, cascade3, cascade4, folder):
    if(folder[-1] != "/"):
        folder = folder + "/"
    files = [join(folder,f) for f in listdir(folder) if isfile(join(folder,f))]

    windowName = None
    totalCount = 0
    for f in files:
        img = cv2.imread(f, 1)

        s = 700
        height, width = img.shape[:2]
        if height>s or width>s:
            if height>width:
                dim = (round(width*(s/height)), s)
            else:
                dim = (s, round(height*(s/width)))
            img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
            # print(img.shape)

        if type(img) is np.ndarray:
            lCnt = detect(img, cascade1, cascade2, cascade3, cascade4)
            totalCount += lCnt
            if windowName != None:
                cv2.destroyWindow(windowName)
            windowName = f
            cv2.namedWindow(windowName, cv2.WINDOW_AUTOSIZE)
            # cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
            # cv2.resizeWindow(windowName, 400, 400);
            # cv2.moveWindow(windowName, 20, 20);  
            cv2.imshow(windowName, img)
            cv2.waitKey(0)
    return totalCount

def runonVideo(face_cascade, face_cascade_default, eyes_cascade, eyes_cascade_glasses):
    videocapture = cv2.VideoCapture(0)
    if not videocapture.isOpened():
        print("Can't open default video camera!")
        exit()

    windowName = "Live Video"
    showlive = True
    while(showlive):
        ret, frame = videocapture.read()

        if not ret:
            print("Can't capture frame")
            exit()

        detect(frame, face_cascade, face_cascade_default, eyes_cascade, eyes_cascade_glasses)
        cv2.imshow(windowName, frame)
        if cv2.waitKey(30) >= 0:
            showlive = False
    
    # outside the while loop
    videocapture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # check command line arguments: nothing or a folderpath
    if len(sys.argv) != 1 and len(sys.argv) != 2:
        print(sys.argv[0] + ": got " + str(len(sys.argv) - 1) 
              + " arguments. Expecting 0 or 1:[image-folder]")
        exit()

    # load pretrained cascades
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
    face_cascade_default = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    # face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    eye_cascade_glasses = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')
    # eye_cascade = cv2.CascadeClassifier('parojos.xml')

    if(len(sys.argv) == 2): # one argument
        folderName = sys.argv[1]
        detections = run_on_folder(face_cascade, face_cascade_default, eye_cascade, eye_cascade_glasses, folderName)
        print("Total of ", detections, "detections")
    else: # no arguments
        runonVideo(face_cascade, face_cascade_default, eye_cascade, eye_cascade_glasses)

