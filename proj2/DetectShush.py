import numpy as np
import cv2
import os
from os import listdir
from os.path import isfile, join
import sys

def detectShush(frame, location, face_m, face_h, ypos, ROI, cascade):
    mouths = cascade.detectMultiScale(ROI, 1.1, 7, 0, (15, 15)) 
    res_m = len(mouths)

    for (mx, my, mw, mh) in mouths:
        mx += location[0]
        my += location[1]
        m_x_medium = (mx+mx+mw)/2

        if abs(m_x_medium - face_m) > mw/3 or abs(ypos - (my+mh)) > int(face_h/5):
            res_m-=1
        else:
            cv2.rectangle(frame, (mx, my), (mx+mw, my+mh), (0, 0, 255), 2)

    return res_m == 0

def detect(frame, faceCascade, mouthsCascade):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    gray_frame = cv2.equalizeHist(gray_frame)
    # gray_frame = cv2.medianBlur(gray_frame, 5)

    faces = faceCascade.detectMultiScale(gray_frame, 1.15, 5, 0|cv2.CASCADE_SCALE_IMAGE, (50, 50))

    if len(faces)==0:
        faces = faceCascade.detectMultiScale(gray_frame, 1.1, 5, 0|cv2.CASCADE_SCALE_IMAGE, (40, 40))

    detected = 0

    for m in faces:
        x, y, w, h = m[0], m[1], m[2], m[3]
        pos = (x, y + int(0.6*h))
        pos_l = (x, y + int(0.6*h))
        mouthROI = gray_frame[(y+int(0.6*h)):y+h, x:x+w]
        mouthROI_l = gray_frame[(y+int(0.6*h)):y+h, x:x+w]
        x_medium = (x+x+w)/2
        y_lowerpos = y + h
        height = h

        if detectShush(frame, pos, x_medium, height, y_lowerpos, mouthROI, mouthsCascade):
            detected += 1
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        else:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return detected

def run_on_folder(cascade1, cascade2, folder):
    if(folder[-1] != "/"):
        folder = folder + "/"
    files =  [join(folder,f) for f in listdir(folder) if isfile(join(folder,f))]
    windowName = None
    totalCnt = 0
    for f in files:
        img = cv2.imread(f)

        s = 700
        height, width = img.shape[:2]


        if height>s or width>s:
            if height>width:
                dim = (round(width*(s/height)), s)
            else:
                dim = (s, round(height*(s/width)))
            img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

        l = 400
        if height<l and width<l:
            if height>width:
                dim = (round(width*(l/height)), l)
            else:
                dim = (l, round(height*(l/width)))
            img = cv2.resize(img, dim, interpolation = cv2.INTER_CUBIC)

        if type(img) is np.ndarray:
            lCnt = detect(img, cascade1, cascade2)
            totalCnt += lCnt
            if windowName != None:
                cv2.destroyWindow(windowName)
            windowName = f
            cv2.namedWindow(windowName, cv2.WINDOW_AUTOSIZE)
            cv2.imshow(windowName, img)
            cv2.waitKey(0)
    return totalCnt

def runonVideo(face_cascade, eyes_cascade):
    videocapture = cv2.VideoCapture(0)
    if not videocapture.isOpened():
        print("Can't open default video camera!")
        exit()

    windowName = "Live Video"
    showframe = True
    while(showframe):
        ret, frame = videocapture.read()

        if not ret:
            print("Can't capture frame")
            break
        detect(frame, face_cascade, eyes_cascade)
        cv2.imshow(windowName, frame)
        if cv2.waitKey(30) >= 0:
            showframe = False
    
    videocapture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # check command line arguments: nothing or a folderpath
    if len(sys.argv) != 1 and len(sys.argv) != 2:
        print(sys.argv[0] + ": got " + str(len(sys.argv) - 1) 
              + " arguments. Expecting 0 or 1:[image-folder]")
        exit()

    # load pretrained cascades
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    # face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
    mouth_cascade = cv2.CascadeClassifier('Mouth.xml')

    if(len(sys.argv) == 2): # one argument
        folderName = sys.argv[1]
        detections = run_on_folder(face_cascade, mouth_cascade, folderName)
        print("Total of ", detections, "detections")
    else: # no arguments
        runonVideo(face_cascade, mouth_cascade)
