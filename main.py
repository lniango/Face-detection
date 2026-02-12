#Source: https://ashadali.medium.com/image-processing-with-opencv-face-detection-on-images-and-videos-2896f54c8caf
# https://www.geeksforgeeks.org/machine-learning/face-recognition-using-artificial-intelligence/

import numpy as np
import cv2
import matplotlib.pyplot as plt
#from cloudinit.distros import fetch

img = cv2.imread("/home/louis/Pictures/Identite.jpg")
print(img.shape) #H W C

#Conversion to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#plt.imshow(gray, cmap='gray')
#plt.show()

#Loading the Haar Cascade classifier
'''
    A Haar cascade is a pre-trained classifier for object detection, 
    often used for face detection. It uses features like edges, lines, and textures to detect faces.
'''
haar = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#Detecting faces
faces = haar.detectMultiScale(gray)
print(faces) # (x, y, width, height)

#Drawing rectanges around faces
cv2.rectangle(img, (168, 165), (168+447, 165+447), (0, 255, 0), 3)
#cv2.rectangle(img, (250, 605), (250+282, 605+282), (255, 0, 0), 3) #Faux positif
#plt.imshow(img)
#plt.show()

cv2.imshow('detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Cropping the detected face
face_crop = img[165:165+447, 168:168+447] #img[y_start:y_end, x_start:x_end]
plt.imshow(face_crop)
cv2.imwrite("male_01.png", face_crop)
#plt.show()