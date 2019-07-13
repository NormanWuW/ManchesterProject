import cv2 as cv
import numpy as np


def add_demo(m1, m2):
    dst = cv.add(m1, m2)
    cv.imshow("add_demo", dst)


def subtract_demo(m1, m2):
    dst = cv.subtract(m1, m2)
    cv.imshow("subtract_demo", dst)


def multiply_demo(m1, m2):
    dst = cv.multiply(m1, m2)
    cv.imshow("multiple_demo", dst)


def divide_demo(m1, m2):
    dst = cv.divide(m1, m2)
    cv.imshow("divide_demo", dst)


def logic_demo(m1, m2):
    dst = cv.bitwise_and(m1, m2)
    dst2 = cv.bitwise_or(m1, m2)
    cv.imshow("logic_demo", dst)
    cv.imshow("logic_demo", dst2)


def contrast_brightness_demo(image, c, b):
    h, w, ch = image.shape
    blank = np.zeros([h, w, ch], image.dtype)
    dst = cv.addWeighted(image, c, blank, 1-c, b)
    cv.imshow("con-bri-demo", dst)


def others(m1, m2):
    M1, dev1 = cv.meanStdDev(m1)
    M2, dev2 = cv.meanStdDev(m2)
    h, w = m1.shape[:2]
    print(M1)
    print(M2)

    print(dev1)
    print(dev2)

    img = np.zeros([h, w], np.uint8)
    m, dev = cv.meanStdDev(img)
    cv.imshow('img', img)
    print(m)
    print(dev)


src1 = cv.imread("D:/FIRST TIME CHICKEN.png")
src2 = cv.imread("D:/SECOND CHICKEN.png")
print(src1.shape)
print(src2.shape)
# cv.namedWindow("input image1", cv.WINDOW_AUTOSIZE)
# cv.imshow("input image1", src1)
# add_demo(src1, src2)
# subtract_demo(src1, src2)
# multiply_demo(src1, src2)
# divide_demo(src1, src2)
# others(src1, src2)
# logic_demo(src1, src2)
cv.imshow("input image2", src2)
contrast_brightness_demo(src2, 1.2, 100)

cv.waitKey(0)
cv.destroyAllWindows()
