from datetime import datetime
import cv2
import math
import time
import imutils
import numpy as np
from matplotlib import pyplot as plt

from obj_tracker.exceptions import TrackerExit


class ObjTracker(object):

    """ Simple object tracker class """

    window_name = 'Object-Tracker'
    scale = 1000  # 4
    contour_color = 0, 0, 255  # , 255, 255
    contour_width = 2
    min_area = 5 * scale
    frame_memory = 2

    font_name = cv2.FONT_HERSHEY_DUPLEX
    compare_method = cv2.cv.CV_COMP_CORREL

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
        blure = cv2.GaussianBlur(grays, (11, 11), 0)
        if prev_frame is None:
            return frame, [blure, ] * self.frame_memory

        prev_frame = prev_frame[:self.frame_memory]
        frame_delta = cv2.absdiff(prev_frame[-1], grays)

        thresh = cv2.threshold(frame_delta, 20, 255, cv2.THRESH_BINARY)[1]
        dil = cv2.dilate(thresh, np.ones((7, 7), np.uint8))

        contours, hierarchy = cv2.findContours(dil.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) < self.min_area:
                continue
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), self.contour_color, self.contour_width)

            sub_img = frame[y:y + h, x:x + w]
            x = self.hist_compare(sub_img, prev_frame[-1][y:y + h, x:x + w])
            # self.draw_hist(sub_img)
            # return sub_img, blure

        return frame, blure

    def draw_hist(self, img):
        for index, color in enumerate(('b', 'g', 'r')):
            histr = cv2.calcHist([img], [index], None, [256], [0, 256])
            plt.plot(histr, color=color)
            plt.xlim([0, 256])
        plt.show()

    def plot_hist(self, histr):
        plt.plot(histr)
        plt.show()

    def fps(self, fps):
        # 0 - frame numbers
        # 1 - fps value
        # 2 - current time
        if not (fps[0] % 10):
            fps_val = fps[0] / (time.time() - fps[2])

            fps[0] = 0
            fps[1] = fps_val
            fps[2] = time.time()

    def hist_compare(self, img_a, img_b):
        """ Compare histograms """

        channels = [0]
        hista = cv2.calcHist([img_a], channels, None, [256], [0, 256])
        histb = cv2.calcHist([img_b], channels, None, [256], [0, 256])

        # self.plot_hist(hista)
        # self.plot_hist(histb)

        # hista = cv2.cv.NormalizeHist(hista, 100.)
        # histb = cv2.cv.NormalizeHist(histb, 100.)


        return cv2.compareHist(hista, histb, self.compare_method)

    def add_text(self, frame, fps):
        """ insert text to frame """
        offset_x, offset_y = 10, 30

        cv2.putText(
            frame, 'fps: %.2f' % fps[1], (offset_x, offset_y), self.font_name, 1, self.contour_color)
        cv2.putText(
            frame, 'time: %s' % datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            (offset_x, len(frame) - offset_y), self.font_name, 1, self.contour_color)

    def run(self):
        frame, prev_frame = self.moution_detect(None)
        # pre fps
        fps = [0, 24, time.time()]
        while True:
            fps[0] += 1
            self.fps(fps)
            self.wait_key()

            frame, tmp_frame = self.moution_detect(prev_frame)
            self.add_text(frame, fps)

            # old frames list
            prev_frame.insert(0, tmp_frame)

            self.show_frame(frame)

    def do(self):
        try:
            self.run()
        except TrackerExit:
            pass


if __name__ == "__main__":
    ObjTracker().do()
