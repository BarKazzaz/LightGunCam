import numpy as np
import cv2 as cv
import torch
from matplotlib import pyplot as plt

im = cv.imread('tv.jpg')
assert im is not None, "file could not be read, check with os.path.exists()"
imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)

ret, thresh = cv.threshold(imgray, 127, 255, 0)
contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
cv.drawContours(im.copy(), contours, -1, (0,255,0), 3)

# contours[2825] is outer tv
# contours[2826] is inner tv

## this will show a rectangle at the middle of the screen
# plt.imshow(
#   cv.rectangle(im.copy(), 
#     [(a+a+b)//2-10 for a,b in zip(cv.boundingRect(contours[2826])[0:2], cv.boundingRect(contours[2826])[2:])],
#     [(a+a+b)//2+10 for a,b in zip(cv.boundingRect(contours[2826])[0:2], cv.boundingRect(contours[2826])[2:])],
#     (255,0,255))
# )

plt.imshow(cv.cvtColor(im, cv.COLOR_BGR2RGB))
plt.show()

## maybe a way to filter out contours?
## I can make sure we deal with rectangular shapes by looking at len(approx)
# approx = cv.approxPolyDP(contours[0], 0.1, True)
# x, y, w, h = cv.boundingRect(approx)
# x_mid = (x + w) // 2
# y_mid = (y + h) // 2

## another option might be to just get the TV position according to yolo
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).autoshape()  # for PIL/cv2/np inputs and NMS

# results = model(im, size=640)  
# results.print()  # print results to screen
# results.show()  # display results
print('done')