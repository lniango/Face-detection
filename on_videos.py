import numpy as np
import cv2
from skimage.feature import hog
from skimage import data, exposure
import matplotlib.pyplot as plt

def face_detect_haar(img):
    haar = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = haar.detectMultiScale(gray, 1.3, 5)

    for x, y, w, h in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 255), 2)

    return img


if __name__ == '__main__':
    #Capturing video input
    cap = cv2.VideoCapture(0) #webcam
    #cap = cv2.VideoCapture("Identite.jpg")
    # Processing each frame
    while True:
        ret, frame = cap.read()
        if ret == False:
            break

        #frame = face_detect_haar(frame)
        frame = face_detect_haar(frame)
        cv2.imshow('Detect', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()



'''
Extending the Code:
    Combine with tracking algorithms for smoother real-time face tracking.
'''