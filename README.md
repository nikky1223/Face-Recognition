# Face-Recognition
#detecting face from images or by using cam


#importing libraries
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

#function to convert the RGB(color image) to BGR image for face recognition
def convertToRGB(img):
    return cv.cvtColor(img, cv.COLOR_BGR2RGB)

#importing cascade file for front face 
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

#getting images from desktop
image=cv.imread('image path/imagename')

#converting image to gray image
grayimage=cv.cvtColor(image,cv.COLOR_BGR2GRAY)

#detecting the faces by using the cascade classifier
faces=face_cascade.detectMultiScale(grayimage,scaleFactor=1.15,minNeighbors=5)

#drawing a rectangle around the face in the core image
for (x, y, w, h) in faces:
    cv.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 3)

#showing the image
plt.imshow(convertToRGB(image))
print('no of faces:')
print(len(faces))
#printing no of images
