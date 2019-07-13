import cv2 as cv
import numpy as np


def lapalian_demo(image):
    dst = cv.Laplacian(image, cv.CV_32F)
    # lpls = cv.convertScaleAbs(dst)
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    dst = cv.filter2D(image, cv.CV_32F, kernel=kernel)
    lpls = cv.convertScaleAbs(dst)
    cv.imshow("lapalian_demo", lpls)


def sobel_demo(image):
    # grad_x = cv.Sobel(image, cv.CV_32F, 1, 0)
    grad_x = cv.Scharr(image, cv.CV_32F, 0, 1)
    grad_y = cv.Scharr(image, cv.CV_32F, 0, 1)
    grandx = cv.convertScaleAbs(grad_x)
    grandy = cv.convertScaleAbs(grad_y)
    cv.imshow("gradient-x", grandx)
    cv.imshow("gradient-y", grandy)

    gradxy = cv.addWeighted(grandx, 0.5, grandy, 0.5, 0)
    cv.imshow("gradientxy", gradxy)


src = cv.imread("D:/FIRST TIME CHICKEN.png")
cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
cv.imshow("input image", src)
# sobel_demo(src)
lapalian_demo(src)
cv.waitKey(0)
cv.destroyAllWindows()