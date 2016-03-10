import time
import cv2
import imutils
import numpy as np

from copy import copy
from imutils import paths

refPt = []
frame_N = 0
frame = None

def click_and_calck(event, x, y, flags, param):
    # grab references to the global variables
    global refPt

    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
 
    elif event == cv2.EVENT_LBUTTONUP:
        refPt.append((x, y))
 
        # draw a rectangle around the region of interest
        cv2.rectangle(frame, refPt[0], refPt[1], (0, 255, 0), 2)
        print (frame_N, (x, y))
        cv2.imshow('image', frame)

cv2.namedWindow('image')
cv2.setMouseCallback('image', click_and_calck)

capture = cv2.VideoCapture('data/cam-01.avi')


while True:

    grabbed, frame = capture.read()
    frame_N += 1

    if not grabbed:
        break

    frame = imutils.resize(frame, width=1200)

    cv2.imshow('image', frame)

    key = cv2.waitKey(10) & 0xFF

    if key == ord('c'):
        break
    elif key == ord('p'):
        time.sleep(2)

cv2.destroyWindow('image')
capture.release()
