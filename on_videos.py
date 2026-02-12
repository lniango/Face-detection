import numpy as np
import cv2

haar = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def face_detect(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = haar.detectMultiScale(gray, 1.3, 5)

    for x, y, w, h in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 255), 2)

    return img


#Capturing video input
cap = cv2.VideoCapture(0) #webcam
#Processing each frame
while True:
    ret, frame = cap.read()
    if ret == False:
        break

    frame = face_detect(frame)
    cv2.imshow('Detect', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

'''
Extending the Code:
    Combine with tracking algorithms for smoother real-time face tracking.
'''