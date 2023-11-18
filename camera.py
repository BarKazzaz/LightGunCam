import numpy as np
import cv2 as cv
import torch
from matplotlib import pyplot as plt

thickness = 5
class COLORS:
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)

def getTvRectangle(img):
    MIN = 0
    MAX = 80
    imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # commented cause min is 0 in my case
    # ret, min_thresh = cv.threshold(imgray, MIN, 255, 0) 
    ret, max_thresh = cv.threshold(imgray, MAX, 255, 200)
    # no need for min as said
    # in_range = np.bitwise_and(min_thresh, np.bitwise_not(max_thresh)) 
    in_range = np.bitwise_not(max_thresh)
    contours, hierarchy = cv.findContours(in_range, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        epsilon = 0.1 * cv.arcLength(cnt, True)
        approx = cv.approxPolyDP(cnt, epsilon, True)
        if len(approx) != 4:
            continue
        x, y, w, h = cv.boundingRect(approx)
        if w < 200 or h < 200:
            continue
        return (x, y), (x + w, y + h)

def getRectCenter(rect):
    p1, p2 = rect
    return ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)

def getCameraAim(img):
    return img.shape[1]//2, img.shape[0]//2 # shape is inverted :(

def main():
    name = 'our2.jpg'
    img = cv.imread(name)
    tv_rect = getTvRectangle(img)
    tv_center = getRectCenter(tv_rect)
    camera_aim = getCameraAim(img)
    result = cv.rectangle(img.copy(), tv_rect[0], tv_rect[1], COLORS.GREEN, thickness)
    result2 = cv.rectangle(result.copy(), (tv_center[0]-10, tv_center[1]-10), (tv_center[0]+10, tv_center[1]+10), COLORS.RED, thickness)
    result3 = cv.line(result2.copy(), (camera_aim[0], camera_aim[1]), (camera_aim[0], camera_aim[1]+10), COLORS.BLUE, thickness)
    result4 = cv.line(result3.copy(), (camera_aim[0]-5, camera_aim[1]+5), (camera_aim[0]+5, camera_aim[1]+5), COLORS.BLUE, thickness)
    plt.imshow(cv.cvtColor(result4, cv.COLOR_BGR2RGB))

if __name__ == '__main__':
    main()
