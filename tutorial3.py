import cv2 as cv
import numpy as np


def extract_object_demo():
    capture = cv.VideoCapture("D:/killercity.mkv")
    while True:
        ret, frame = capture.read()
        if ret == 0:
            break
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        lower_hsv = np.array([37, 43, 46])
        upper_hsv = np.array([77, 255, 255])
        mask = cv.inRange(hsv, lowerb=lower_hsv, upperb=upper_hsv)
        dst = cv.bitwise_and(frame, frame, mask=mask)
        cv.imshow("video", frame)
        cv.imshow("mask", dst)
        c = cv.waitKey(40)
        if c == 27:
            break


def color_space_demo(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    cv.imshow("gray", gray)
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    cv.imshow("hsv", hsv)
    yuv = cv.cvtColor(image, cv.COLOR_BGR2YUV)
    cv.imshow("yuv", yuv)


src = cv.imread("D:/UoM/project/dataset/2000m/2019-06-11_154547.png")
cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
cv.imshow("input image", src)

b, g, r = cv.split(src)
# cv.imshow("blue", b)
# cv.imshow("green", g)
# cv.imshow("red", r)

src[:, :, 2] = 255
src = cv.merge([b, g, r])
cv.imshow("change image", src)

# extract_object_demo()
cv.waitKey(0)
cv.destroyAllWindows()
