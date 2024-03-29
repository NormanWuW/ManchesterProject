import cv2 as cv
import numpy as np


def pyramid_demo(image):
    level = 3
    temp = image.copy()
    pyramid_image = []
    for i in range(level):
        dst = cv.pyrDown(temp)
        pyramid_image.append(dst)
        cv.imshow("pyramid_down_"+str(i), dst)
        temp = dst.copy()
    return pyramid_image


def lapalian_demo(image):
    pyramid_images = pyramid_demo(image)
    level = len(pyramid_images)
    for i in range(level-1, -1, -1):
        if (i-1) < 0:
            expend = cv.pyrUp(pyramid_images[i], dstsize=image.shape[:2])
            lpls = cv.subtract(image, expend)
            cv.imshow("lapalian_down_" + str(i), lpls)
        else:
            expend = cv.pyrUp(pyramid_images[i], dstsize=pyramid_images[i-1].shape[:2])
            lpls = cv.subtract(pyramid_images[i-1], expend)
            cv.imshow("lapalian_down_"+str(i), lpls)


src = cv.imread("D:/FIRST TIME CHICKEN.png")
cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
cv.imshow("input image", src)
# lapalian_demo(src)
cv.waitKey(0)
cv.destroyAllWindows()