import cv2
import math
import imutils
import numpy as np

from obj_tracker.exceptions import TrackerExit


class ObjTracker(object):

    """ Simple object tracker class """

    window_name = 'Object-Tracker'
    scale = 1000  # 4
    contour_color = 0, 0, 255  # , 255, 255
    contour_width = 2
    min_area = 5 * scale

    timeout = 10 # ms
    ANYKEY = True

    DRAW_RECT = False

    def __init__(self):
        cv2.namedWindow(self.window_name, cv2.CV_WINDOW_AUTOSIZE)
        self._init_vebcam()

    def _init_vebcam(self):
        """ Initialize vebcam stream """
        self.capture = cv2.VideoCapture(0)

    def _from_vebcam(self):
        """ Read img from vebcam """
        grabbed, frame = self.capture.read()
        frame = cv2.flip(frame, 1)
        return frame

    def read_frame(self):
        return self._from_vebcam()

    def show_frame(self, frame):
        if frame is not None:
            cv2.imshow(self.window_name, frame)

    def wait_key(self):
        """ exit btns """
        # each timeout in ms
        key = cv2.waitKey(self.timeout)
        # if q was pressed exit
        if key & 255 == ord('q') or (self.ANYKEY and (key != -1)):
            cv2.destroyWindow(self.window_name)
            self.capture.release()
            raise TrackerExit()



    def moution_detect(self, prev_frame):
        """ Detect moution on stream """
        frame = self.read_frame()
        # make grayscale frame
        frame = imutils.resize(frame, width=self.scale)
        grays = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blure = cv2.GaussianBlur(grays, (1, 1), 0)
        if prev_frame is None:
            return frame, blure

        frame_delta = cv2.absdiff(prev_frame, grays)
        thresh = cv2.threshold(frame_delta, 20, 255, cv2.THRESH_BINARY)[1]
        dil = thresh
        dil = cv2.dilate(thresh, np.ones((27, 27), np.uint8))

        contours, hierarchy = cv2.findContours(dil.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) < self.min_area:
                continue
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), self.contour_color, self.contour_width)

        return frame, blure


    def run(self):
        frame, prev_frame = self.moution_detect(None)
        while True:
            self.wait_key()
            frame, prev_frame = self.moution_detect(prev_frame)
            self.show_frame(frame)


    def do(self):
        try:
            self.run()
        except TrackerExit:
            pass


if __name__ == "__main__":
    ObjTracker().do()
