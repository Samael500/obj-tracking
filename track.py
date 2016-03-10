from mftracker import *
import cv2
cap = cv2.VideoCapture("data/cam-01.avi")
_, img = cap.read()
_, img = cap.read()

bb = [1165, 590, 1167, 597]
# cv2.imshow("image", img)
# cv2.waitKey(0)

mftrack("data/cam-01.avi", bb)
