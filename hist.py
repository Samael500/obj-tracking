import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('data/target.png')

# histt = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
# cv2.normalize(histt, histt).flatten()

color = ('b','g','r')
for i, col in enumerate(color):
    histr = cv2.calcHist([img],[i],None,[256],[0,256])
    cv2.normalize(histr, histr).flatten()

    plt.plot(histr,color = col)
    plt.xlim([0,256])
plt.show()
