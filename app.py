import cv2
from random import randrange as r

trainedData=cv2.CascadeClassifier('haarcascade.xml')

webcam=cv2.VideoCapture(0)

while True:
 success,frame=webcam.read()

 grayimg=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

 faceCoordinates=trainedData.detectMultiScale(grayimg)

 for x,y,w,h in faceCoordinates:
    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)

 cv2.imshow('window',frame)
 key=cv2.waitKey(1)
 if(key==81 or key==113):
    break

webcam.release()

img=cv2.imread('b.jpg')

grayimg=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

faceCoordinates=trainedData.detectMultiScale(grayimg)

for x,y,w,h in faceCoordinates:
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)

cv2.imshow('window',img)
cv2.waitKey()

print('End of program')