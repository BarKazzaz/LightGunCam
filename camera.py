import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

im = cv.imread('tv.jpg')
assert im is not None, "file could not be read, check with os.path.exists()"
imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)

ret, thresh = cv.threshold(imgray, 127, 255, 0)
contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
cv.drawContours(im, contours, -1, (0,255,0), 3)

plt.imshow(cv.cvtColor(im, cv.COLOR_BGR2RGB))
plt.show()