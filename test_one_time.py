import cv2 as cv
import numpy as np


def building_detect():
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    building_detector = cv.CascadeClassifier("D:/UoM/project/opencv3.4/opencv/sources/data/haarcascades/cascade.xml")
    buildings = building_detector.detectMultiScale(gray, 1.02, 5)
    for x, y, w, h in buildings:
        cv.rectangle(src, (x, y), (x+w, y+h), (0, 0, 255), 1)
    cv.imshow("result", src)


src = cv.imread("D:/UoM/project/dataset/1750m_positive/17.jpg")
cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
cv.namedWindow("result", cv.WINDOW_AUTOSIZE)
cv.imshow("input image", src)
building_detect()
cv.waitKey(0)
cv.destroyAllWindows()
