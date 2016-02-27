from datetime import datetime
import cv2
import math
import time
import imutils
from matplotlib import pyplot as plt

from obj_tracker.exceptions import TrackerExit

from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import cv2


class ObjTracker(object):

    """ Simple object tracker class """

    VEBCAM = False

    window_name = 'Object-Tracker'
    scale = 600 # 4
    contour_color = 0, 0, 255  # , 255, 255
    contour_width = 2
    min_area = 5 * scale
    frame_memory = 2

    font_name = cv2.FONT_HERSHEY_DUPLEX
    compare_method = cv2.cv.CV_COMP_CORREL

    timeout = 10  # ms
    ANYKEY = True

    DRAW_RECT = False

    skipframe = 64

    def __init__(self):
        cv2.namedWindow(self.window_name, cv2.CV_WINDOW_AUTOSIZE)
        self._init_capture('data/cam-01.mp4')
        # initialize the HOG descriptor/person detector
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def _init_capture(self, path=0):
        """ Initialize vebcam or fileobj video stream """
        self.capture = cv2.VideoCapture(path)

    def _read_frame(self):
        """ Read img from capture """
        for i in range(self.skipframe):
            grabbed, frame = self.capture.read()
        if self.VEBCAM:
            frame = cv2.flip(frame, 1)

        frame = imutils.resize(frame, width=self.scale)
        return frame

    def read_frame(self):
        return self._read_frame()

    def show_frame(self, frame):
        if frame is not None:
            cv2.imshow(self.window_name, frame)

    def moution_detect(self, frame, prev_frame):
        """ Detect moution on stream """
        # make grayscale frame
        grays = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blure = cv2.GaussianBlur(grays, (11, 11), 0)

        frame_delta = cv2.absdiff(blure, grays)

        thresh = cv2.threshold(frame_delta, 20, 255, cv2.THRESH_BINARY)[1]
        dil = cv2.dilate(thresh, np.ones((7, 7), np.uint8))

        contours, hierarchy = cv2.findContours(dil.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) < self.min_area:
                continue
            (x, y, w, h) = cv2.boundingRect(contour)
            sub_img = frame[y:y + h, x:x + w]
            img = self.people_find(sub_img)

            frame[y:y + h, x:x + w] = img


        return frame, blure

    def wait_key(self):
        """ exit btns """
        # each timeout in ms
        key = cv2.waitKey(self.timeout)
        # if q was pressed exit
        if key & 255 == ord('q') or (self.ANYKEY and (key != -1)):
            cv2.destroyWindow(self.window_name)
            self.capture.release()
            raise TrackerExit()

    def people_find(self, image):
        rects, weights = self.hog.detectMultiScale(
            image, winStride=(4, 4), padding=(4, 4), scale=1.05
        )
 
        # draw the original bounding boxes
        # for (x, y, w, h) in rects:
        #     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
 
        # apply non-maxima suppression to the bounding boxes using a
        # fairly large overlap threshold to try to maintain overlapping
        # boxes that are still people
        rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
        pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
 
        # draw the final bounding boxes
        for (xA, yA, xB, yB) in pick:
            cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)

        return image

    def run(self):
        prev_frame = self.read_frame()
        while True:
            frame = self.read_frame()
            self.wait_key()
            oframe = self.moution_detect(frame, prev_frame)
            self.show_frame(oframe)
            prev_frame = frame

    def do(self):
        try:
            self.run()
        except TrackerExit:
            pass


if __name__ == "__main__":
    ObjTracker().do()
