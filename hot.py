import math
import cv2
import imutils
import numpy as np

from copy import copy
from imutils import paths
# from skimage.measure import structural_similarity as ssim

from obj_tracker.exceptions import TrackerExit

from x import xall

no_euqlid = True

for i in range(471, 1898):
    if i in xall:
        last = xall[i]
    else:
        xall[i] = last

def euqlid(A, B):
    if B is None:
        return 0
    return math.sqrt((A[0] - B[0]) ** 2 + (A[1] - B[1]) ** 2)

frame_N = 0
savex = {}

def click_and_calck(event, x, y, flags, param):
    # grab references to the global variables
    if event == cv2.EVENT_LBUTTONDOWN:
        for nm, im in savex.items():
            cv2.imwrite('data/frames/%d-%s.jpg' % (frame_N, nm), im)


class ObjTracker(object):

    """ Simple object tracker class """

    VEBCAM = False

    window_name = 'Object-Tracker'
    scale = 1200  # 4
    contour_color = 0, 0, 255
    target_color = 0, 255, 0
    contour_width = 2
    min_area = .7 * scale
    frame_memory = 2

    # detect
    min_tresh = .9
    max_dist = 50
    center = None#1165, 590
    track_window = 1165, 590, 400, 800

    font_name = cv2.FONT_HERSHEY_DUPLEX
    # compare_method = cv2.CV_COMP_CORREL

    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 10)

    timeout = 10  # ms
    ANYKEY = True

    DRAW_RECT = False

    skipframe = 1

    def __init__(self):
        cv2.namedWindow(self.window_name)#, cv2.CV_WINDOW_AUTOSIZE)
        self._init_capture('data/input.avi')  # mp4')
        self.target = cv2.imread('data/target.png')
        self.target2 = cv2.imread('data/target2.png')
        # self.writer = cv2.VideoWriter(
        #     filename='data/output.avi',
        #     fourcc=cv2.VideoWriter_fourcc(*'XVID'),
        #     fps=30.0, frameSize=(1200, 843))
        # cv2.setMouseCallback(self.window_name, click_and_calck)

    def _init_capture(self, path=0):
        """ Initialize vebcam or fileobj video stream """
        self.capture = cv2.VideoCapture(path)

    def _read_frame(self):
        """ Read img from capture """
        global frame_N, writer
        for i in range(self.skipframe):
            frame_N += 1
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
            # self.writer.write(frame)

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
        x, contours, hierarchy = cv2.findContours(dil, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        clean = frame.copy()

        savex['1-gray'] = grays[0]
        savex['2-blure'] = blure[0]
        savex['3-delta'] = frame_delta
        savex['4-thresh'] = thresh
        savex['5-dil'] = dil.copy()

        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)
            if cv2.contourArea(contour) < self.min_area:
                continue
            # image cmp
            cv2.rectangle(
                frame, (x, y), (x + w, y + h),
                self.target_color if self.compare(
                    clean[y: y + h, x: x + w], (x, y, w, h)) else self.contour_color,
                self.contour_width)
            cv2.rectangle(dil, (x, y), (x + w, y + h), (255, 255, 255), self.contour_width)

        savex['6-countours'] = dil

        if self.center:
            (x, y, w, h) = self.center

            roi = clean[y: y + h, x: x + w]
            hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
            roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
            cv2.normalize(roi_hist,roi_hist).flatten()
            
            hsv = cv2.cvtColor(clean, cv2.COLOR_BGR2HSV)
            dst = cv2.calcBackProject([hsv],[0], roi_hist,[0,180],1)

            term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1, 1 )
            ret, center = cv2.CamShift(grays[1], self.center, term_crit)
            x,y,w,h = center
            cv2.rectangle(frame, (x,y), (x+w,y+h), 255,2)

            # self.target = clean[y: y + h, x: x + w]

        #     # blank_image = np.zeros((h, w, 3), np.uint8)
        #     # frame[y: y + h, x: x + w] = self.people_find(frame[y: y + h, x: x + w])

        return frame

    def comparex(self, center):
        crd = xall.get(frame_N)
        if crd:
            return (center[0] < crd[0] < center[0] + center[2]) and (center[1] < crd[1] < center[1] + center[3])
        return False

    def compare(self, image, center):
        # image histogram
        hista = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        cv2.normalize(hista, hista).flatten()
        # target histogram
        histt = cv2.calcHist([self.target], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        cv2.normalize(histt, histt).flatten()
        # compare result
        compare = cv2.compareHist(histt, hista, cv2.HISTCMP_CORREL)
        if compare > self.min_tresh and euqlid(center, self.center) < self.max_dist or self.comparex(center):
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
        prev_frame = [self.read_frame()]
        while True:
            frame = self.read_frame()
            oframe = self.moution_detect(frame.copy(), prev_frame[0].copy())
            self.show_frame(oframe)
            prev_frame.append(frame)
            self.wait_key()
            if len(prev_frame) > 11:
                prev_frame.pop(0)

    def do(self):
        try:
            self.run()
        except TrackerExit:
            pass
        finally:
            cv2.destroyWindow(self.window_name)
            self.capture.release()
            # self.writer.release()

if __name__ == "__main__":
    ObjTracker().do()
