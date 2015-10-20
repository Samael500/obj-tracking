import cv2
import math
import imutils
import numpy as np


class ObjTracker(object):

    """ Simple object tracker class """

    window_name = 'Object-Tracker'
    scale_down = 1  # 4
    contour_color = 0, 0, 255
    contour_width = 2
    min_area = 50

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

    def _hight_contrast(self, orig_img):
        """ Create contrasted img with color block """
        img = cv2.GaussianBlur(orig_img, (5, 5), 0)
        img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2HSV)
        # minified img for high performance
        img = cv2.resize(img, (len(orig_img[0]) / self.scale_down, len(orig_img) / self.scale_down))
        return img

    def _select_color_rage(self, img):
        """ Hightlight bright colors """
        binary = cv2.inRange(img, np.array([0, 150, 0], np.uint8), np.array([5, 255, 250], np.uint8))
        dilation = np.ones((15, 15), np.uint8)
        return cv2.dilate(binary, dilation)

    def _find_contours(self, binary):
        """ Find all contours in given binary img """
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def _largest_contour(self, contours):
        """ find the largest countour """
        max_area = 0
        largest_contour = None
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > max_area:
                max_area = area
                largest_contour = contour
        return largest_contour

    def _draw_countour(self, orig_img, largest_contour):
        if largest_contour is None:
            return orig_img

        moment = cv2.moments(largest_contour)

        if moment['m00'] > 1000 / self.scale_down:

            if self.DRAW_RECT:
                rect = cv2.minAreaRect(largest_contour)
                rect = (
                    # change rect to actual scale size
                    (rect[0][0] * self.scale_down, rect[0][1] * self.scale_down),
                    (rect[1][0] * self.scale_down, rect[1][1] * self.scale_down),
                    rect[2]
                )
                box = cv2.cv.BoxPoints(rect)
                box = np.int0(box)

                cv2.drawContours(orig_img, (box, ), 0, self.contour_color, self.contour_width)

            else:
                (x, y), radius = cv2.minEnclosingCircle(largest_contour)
                center = (int(x), int(y))
                radius = int(radius)

                cv2.circle(orig_img, center, radius, self.contour_color, self.contour_width)
                cv2.circle(orig_img, center, 1, self.contour_color, self.contour_width)

        return orig_img

    def color_track(self):
        """ Track red block """
        orig_img = self._from_vebcam()
        img = self._hight_contrast(orig_img)
        binary = self._select_color_rage(img)
        contours = self._find_contours(binary)
        largest_contour = self._largest_contour(contours)
        drawed_img = self._draw_countour(orig_img, largest_contour)
        return drawed_img

    def _grayscale(self, frame):
        """ Grayscaled img """
        # resize for optimize performance
        frame = imutils.resize(frame, width=500)
        # make grayscale - color is not importan
        grays = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        grays = cv2.GaussianBlur(grays, (21, 21), 0)
        return frame, grays

    def _delta(self, first_frame, grays):
        frame_delta = cv2.absdiff(first_frame, grays)
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
        return thresh

    def _motion_countor(self, frame, thresh):
        thresh = cv2.dilate(thresh, None, iterations=2)
        contours, th = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) < self.min_area:
                continue
    
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), self.contour_color, self.contour_width)
            # text = "Occupied"



    def motion_track(self, first_frame):
        """ Track motion block """
        frame = self._from_vebcam()
        frame, grays = self._grayscale(frame)

        if first_frame is None:
            return frame, grays

        thresh = self._delta(first_frame, grays)
        self._motion_countor(frame, thresh)

        # drawed_img = orig_img
        return thresh, grays


    def run(self):
        first_frame = None
        while True:

            if cv2.waitKey(50) != -1:
                cv2.destroyWindow(self.window_name)
                self.capture.release()
                break

            # drawed_img = self.color_track()
            drawed_img, first_frame = self.motion_track(first_frame)

            if drawed_img is not None:
                cv2.imshow(self.window_name, drawed_img)


if __name__ == "__main__":
    ObjTracker().run()
