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
''''
def face_detect_hog(img): #histogram of gradient
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features, hog_image = hog(img,
                              orientations=8,
                              pixels_per_cell=(16, 16),
                              cells_per_block=(2, 2),
                              visualize=True,
                              channel_axis=-1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    ax1.axis('off')
    ax1.imshow(img, cmap=plt.cm.gray)
    ax1.set_title('Input image', fontsize=15)

    ax2.axis('off')
    ax2.imshow(hog_image, cmap=plt.cm.gray)
    ax2.set_title('Histogram of Oriented Gradients', fontsize=15)
    plt.show()
'''
# https://pyimagesearch.com/2021/04/19/face-detection-with-dlib-hog-and-cnn/
def face_detect_hog(img):
    detector = dlib.get_frontal_face_detector()

    #image = cv2.imread(img)
    image = imutils.resize(img, width=600)
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
    #Capturing video input
    cap = cv2.VideoCapture(0) #webcam
    #cap = cv2.VideoCapture("Identite.jpg")
    # Processing each frame
    while True:
        ret, frame = cap.read()
        if ret == False:
            break

        #frame = face_detect_haar(frame)
        frame = face_detect_hog(frame)
        cv2.imshow('Detect', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()



'''
Extending the Code:
    Combine with tracking algorithms for smoother real-time face tracking.
'''