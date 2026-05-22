import numpy as np
import cv2
from skimage.feature import hog
from skimage import data, exposure
import matplotlib.pyplot as plt
import dlib
#convert_and_trim_bb
from helpers import convert_and_trim_bb
import imutils

#############################
#HOG and SVM
#############################
#https://www.geeksforgeeks.org/computer-vision/histogram-of-oriented-gradients/

# https://pyimagesearch.com/2021/04/19/face-detection-with-dlib-hog-and-cnn/
def face_detect_hog(img, detector): 
    if img is None:
        print("Frame is None")
        return img
    
    if not isinstance(img, np.ndarray):
        print("Not numpy:", type(img))
        return img

    if img.size == 0:
        print("Empty frame")
        return img
    #print("frame type:", type(img))
    #print("shape:", getattr(img, "shape", None))


    #image = cv2.imread(img)
    image = imutils.resize(img, width=600)
    #image = np.ascontiguousarray(image)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #extract rectangles
    print("performing face detection")
    rects = detector(rgb_image)

    boxes = [convert_and_trim_bb(image, r) for r in rects]
    #loop over the bounding boxes
    for (x, y, w, h) in boxes:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    #cv2.imshow("Face detection", image)
    #cv2.waitKey(0)
    return image

if __name__ == '__main__':
    detector = dlib.get_frontal_face_detector()
    
    #Capturing video input
    cap = cv2.VideoCapture(1) #webcam
    #cap = cv2.VideoCapture("Identite.jpg")
    # Processing each frame
    while True:
        ret, frame = cap.read()
        #if ret == False:
        #   break
        if not ret:
            print("Camera failed")
            continue
        
        if frame is None or frame.size == 0:
            print("Bad frame")
            continue

        #frame = face_detect_haar(frame)
        frame = face_detect_hog(frame, detector)
        cv2.imshow('Detect', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

