import cv2

face_cascade = cv2.cascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.videoCapture(0)
"""capture the camera"""


while True : 
    _, img = cap.read()
    """helps to read each frame of the video"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    """converts captured photos to gray tones"""
    faces = face_cascade.detectMultiScale(gray , 1.1 , 4)
