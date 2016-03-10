import math
import cv2
import imutils
import numpy as np

from imutils import paths
from skimage.measure import structural_similarity as ssim

from obj_tracker.exceptions import TrackerExit

no_euqlid = True

def euqlid(A, B):
    return math.sqrt((A[0] - B[0]) ** 2 + (A[1] - B[1]) ** 2)


class ObjTracker(object):

    """ Simple object tracker class """

    VEBCAM = False

    window_name = 'Object-Tracker'
    scale = 1200  # 4
    contour_color = 0, 0, 255
    target_color = 0, 255, 0
    contour_width = 2
    min_area = 2 * scale
    frame_memory = 2

    # detect
    min_tresh = .85
    max_dist = 60
    center = 1165, 590

    font_name = cv2.FONT_HERSHEY_DUPLEX
    compare_method = cv2.cv.CV_COMP_CORREL

    timeout = 10  # ms
    ANYKEY = True

    DRAW_RECT = False

    skipframe = 3

    def __init__(self):
        cv2.namedWindow(self.window_name, cv2.CV_WINDOW_AUTOSIZE)
        self._init_capture('data/cam-01.avi')  # mp4')
        self.target = cv2.imread('data/target.png')

    def _init_capture(self, path=0):
        """ Initialize vebcam or fileobj video stream """
        self.capture = cv2.VideoCapture(path)

    def _read_frame(self):
        """ Read img from capture """
        for i in range(self.skipframe):
            grabbed, frame = self.capture.read()
            if not grabbed:
                raise TrackerExit()
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
        grays = [cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY) for frm in (frame, prev_frame)]
        # blurr for symplify image
        blure = [cv2.GaussianBlur(frm, (23, 23), 0) for frm in grays]
        # calculate coutours
        frame_delta = cv2.absdiff(*blure)
        # hight contrast
        thresh = cv2.threshold(frame_delta, 20, 255, cv2.THRESH_BINARY)[1]
        dil = cv2.dilate(thresh, np.ones((21, 21), np.uint8))
        # find countours
        contours, hierarchy = cv2.findContours(dil, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)
            if cv2.contourArea(contour) < self.min_area:
                continue
            # image cmp
            cv2.rectangle(
                frame, (x, y), (x + w, y + h),
                self.target_color if self.compare(
                    frame[y: y + h, x: x + w], (x, y)) else self.contour_color,
                self.contour_width)

        #     # blank_image = np.zeros((h, w, 3), np.uint8)
        #     # frame[y: y + h, x: x + w] = self.people_find(frame[y: y + h, x: x + w])

        return frame

    def compare(self, image, center):
        # image histogram
        hista = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hista = cv2.normalize(hista).flatten()
        # target histogram
        histt = cv2.calcHist([self.target], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        histt = cv2.normalize(histt).flatten()
        # compare result
        compare = cv2.compareHist(histt, hista, cv2.cv.CV_COMP_CORREL)
        if compare > self.min_tresh and euqlid(center, self.center) < self.max_dist:
            self.target = image
            self.center = center
            return True
        return False

    def wait_key(self):
        """ exit btns """
        # each timeout in ms
        key = cv2.waitKey(self.timeout)
        # if q was pressed exit
        if key & 255 == ord('q') or (self.ANYKEY and (key != -1)):
            raise TrackerExit()

    def run(self):
        prev_frame = self.read_frame()
        while True:
            frame = self.read_frame()
            oframe = self.moution_detect(frame.copy(), prev_frame.copy())
            self.show_frame(oframe)
            prev_frame = frame
            self.wait_key()

    def do(self):
        try:
            self.run()
        except TrackerExit:
            pass
        finally:
            cv2.destroyWindow(self.window_name)
            self.capture.release()


if __name__ == "__main__":
    ObjTracker().do()
